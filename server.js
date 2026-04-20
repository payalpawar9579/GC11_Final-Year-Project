const express = require("express");
const multer = require("multer");
const dotenv = require("dotenv");
const path = require("path");
const fs = require("fs");
const { spawn } = require("child_process");

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;

// Node 18+ has global fetch. If not, install node-fetch.
const fetchFn = global.fetch || require("node-fetch");

// ===========================
// FOLDERS
// ===========================
const rootUploadDir = path.join(__dirname, "uploads");
const backendUploadDir = path.join(__dirname, "backend", "uploads");
const publicDir = path.join(__dirname, "public");
const templatesDir = path.join(__dirname, "backend", "templates");

for (const dir of [rootUploadDir, backendUploadDir, publicDir, templatesDir]) {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
}

// ===========================
// MIDDLEWARE
// ===========================
app.use(express.json({ limit: "10mb" }));
app.use(express.urlencoded({ extended: true }));

app.use("/public", express.static(publicDir));
app.use("/uploads", express.static(backendUploadDir));
app.use("/backend/uploads", express.static(backendUploadDir));

// ===========================
// MULTER
// ===========================
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, backendUploadDir);
  },
  filename: (req, file, cb) => {
    const ext = path.extname(file.originalname || ".jpg");
    cb(null, `img_${Date.now()}${ext}`);
  }
});

const upload = multer({ storage });

// ===========================
// HELPERS
// ===========================
function getEspIp(req) {
  const ip = String(req.query.ip || req.body.ip || "").trim();
  if (!ip) {
    throw new Error("ESP32 IP is required");
  }
  return ip;
}

async function fetchEspJson(url) {
  const resp = await fetchFn(url, { method: "GET" });
  const text = await resp.text();

  let data = {};
  try {
    data = text ? JSON.parse(text) : {};
  } catch (e) {
    throw new Error(`Invalid JSON from ESP32: ${text}`);
  }

  if (!resp.ok) {
    throw new Error(data.error || `ESP32 request failed with status ${resp.status}`);
  }

  return data;
}

// ===========================
// HEALTH CHECK
// ===========================
app.get("/api/health", (req, res) => {
  res.json({
    ok: true,
    message: "Backend is running"
  });
});

// ===========================
// ESP32 PROXY ROUTES
// ===========================
app.get("/api/esp32/health", async (req, res) => {
  try {
    const ip = getEspIp(req);
    const data = await fetchEspJson(`http://${ip}/health`);
    res.json(data);
  } catch (error) {
    console.error("ESP32 /health proxy error:", error.message);
    res.status(500).json({
      ok: false,
      error: error.message
    });
  }
});

app.get("/api/esp32/data", async (req, res) => {
  try {
    const ip = getEspIp(req);
    const data = await fetchEspJson(`http://${ip}/data`);
    res.json(data);
  } catch (error) {
    console.error("ESP32 /data proxy error:", error.message);
    res.status(500).json({
      ok: false,
      error: error.message
    });
  }
});

// Optional direct image proxy
app.get("/api/esp32/capture", async (req, res) => {
  try {
    const ip = getEspIp(req);
    const resp = await fetchFn(`http://${ip}/capture`);
    if (!resp.ok) {
      throw new Error(`ESP32 capture failed with status ${resp.status}`);
    }
    const arrayBuffer = await resp.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);
    res.setHeader("Content-Type", resp.headers.get("content-type") || "image/jpeg");
    res.send(buffer);
  } catch (error) {
    console.error("ESP32 /capture proxy error:", error.message);
    res.status(500).json({
      ok: false,
      error: error.message
    });
  }
});

// ===========================
// HOME ROUTE
// ===========================
app.get("/", (req, res) => {
  res.sendFile(path.join(templatesDir, "index.html"));
});

// ===========================
// IMAGE ANALYSIS
// ===========================
app.post("/api/analyze-image", upload.single("image"), (req, res) => {
  try {
    console.log("POST /api/analyze-image called");

    if (!req.file) {
      return res.status(400).json({
        success: false,
        error: "No image file received"
      });
    }

    const tempC = req.body.tempC || "0";
    const phValue = req.body.phValue || "0";
    const imagePath = req.file.path;
    const pythonScript = path.join(__dirname, "backend", "predict_one.py");
    const pythonCmd = process.platform === "win32" ? "py" : "python";

    const pythonProcess = spawn(pythonCmd, [
      pythonScript,
      imagePath,
      tempC,
      phValue
    ]);

    let result = "";
    let errorOutput = "";

    pythonProcess.stdout.on("data", (data) => {
      result += data.toString();
    });

    pythonProcess.stderr.on("data", (data) => {
      errorOutput += data.toString();
    });

    pythonProcess.on("close", (code) => {
      if (code !== 0) {
        console.error("Python error:", errorOutput);
        return res.status(500).json({
          success: false,
          error: "Python prediction failed",
          details: errorOutput
        });
      }

      try {
        const parsed = JSON.parse(result);
        return res.json(parsed);
      } catch (parseError) {
        console.error("Invalid JSON from Python:", result);
        return res.status(500).json({
          success: false,
          error: "Invalid response from Python backend",
          raw: result,
          details: errorOutput
        });
      }
    });
  } catch (error) {
    console.error("Error in /api/analyze-image:", error);
    return res.status(500).json({
      success: false,
      error: "Server error while analyzing image"
    });
  }
});

// ===========================
// CHAT API
// ===========================
app.post("/api/chat", async (req, res) => {
  try {
    const { question, live } = req.body || {};

    const temp =
      typeof live?.tempC === "number" ? `${live.tempC.toFixed(2)} °C` : "N/A";
    const phVoltage =
      typeof live?.phVoltage === "number"
        ? `${live.phVoltage.toFixed(3)} V`
        : "N/A";
    const phValue =
      typeof live?.phValue === "number" ? live.phValue.toFixed(2) : "N/A";
    const redness =
      typeof live?.image?.rednessPct === "number"
        ? `${live.image.rednessPct.toFixed(0)}%`
        : "N/A";
    const flake =
      typeof live?.image?.flakePct === "number"
        ? `${live.image.flakePct.toFixed(0)}%`
        : "N/A";
    const bumps =
      typeof live?.image?.bumps === "number" ? live.image.bumps : "N/A";

    const reply =
      `AI backend working.\n` +
      `Question: ${question || "No question provided"}\n` +
      `Temperature: ${temp}\n` +
      `pH Voltage: ${phVoltage}\n` +
      `pH Value: ${phValue}\n` +
      `Redness: ${redness}\n` +
      `Flakiness: ${flake}\n` +
      `Bumps: ${bumps}`;

    return res.json({ reply });
  } catch (error) {
    console.error("Error in /api/chat:", error);
    return res.status(500).json({
      error: "Server error while generating chat reply"
    });
  }
});

// ===========================
// FALLBACK ROUTE
// ===========================
app.use((req, res) => {
  res.sendFile(path.join(templatesDir, "index.html"));
});

// ===========================
// START SERVER
// ===========================
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});