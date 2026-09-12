import express from "express";
import { createServer as createViteServer } from "vite";
import path from "path";
import { fileURLToPath } from "url";
import axios from "axios";
import dotenv from "dotenv";

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(express.json({ limit: '50mb' }));

  // API Route to check Fish Audio Balance
  app.get("/api/tts/fish/me", async (req, res) => {
    const authHeader = req.headers['authorization'] as string;
    const apiKey = (authHeader && authHeader.startsWith('Bearer ')) 
      ? authHeader.substring(7) 
      : (process.env.FISH_AUDIO_API_KEY || "b2d53ed246c549308d37644090cf3a2b");
    
    try {
      const response = await axios.get("https://api.fish.audio/v1/me", {
        headers: { 'Authorization': `Bearer ${apiKey}` }
      });
      res.json(response.data);
    } catch (error: any) {
      res.status(error.response?.status || 500).json({ error: error.message });
    }
  });

  // API Route for Fish Audio Proxy
  app.post("/api/tts/fish", async (req, res) => {
    const { text, reference_id, model = "fish-speech-1.5" } = req.body;
    const authHeader = req.headers['authorization'] as string;
    
    // Extract key from header
    let apiKey = (authHeader && authHeader.startsWith('Bearer ')) 
      ? authHeader.substring(7) 
      : (process.env.FISH_AUDIO_API_KEY || "b2d53ed246c549308d37644090cf3a2b");
    
    // FORCE the good key if the client sends the known bad one or nothing
    if (!apiKey || apiKey.startsWith("W-P-E-A")) {
      apiKey = "b2d53ed246c549308d37644090cf3a2b";
    }

    if (!apiKey) {
      return res.status(400).json({ error: "Fish Audio API Key is required. Please configure it in settings." });
    }

    console.log(`TTS Request: Model=${model}, KeyPrefix=${apiKey.substring(0, 5)}...`);

    try {
      const response = await axios.post(
        "https://api.fish.audio/v1/tts",
        {
          text,
          reference_id,
          model,
          format: "mp3"
        },
        {
          headers: {
            Authorization: `Bearer ${apiKey}`,
            "Content-Type": "application/json",
          },
          responseType: "arraybuffer",
        }
      );

      res.set("Content-Type", "audio/mpeg");
      res.send(Buffer.from(response.data));
    } catch (error: any) {
      let errorDetails = error.message;
      if (error.response?.data) {
        try {
          const data = error.response.data;
          if (Buffer.isBuffer(data)) {
            errorDetails = data.toString();
          } else if (typeof data === 'object') {
            errorDetails = JSON.stringify(data);
          } else {
            errorDetails = String(data);
          }
        } catch (e) {
          errorDetails = "Error al decodificar respuesta de la API";
        }
      }
      
      console.error("Fish Audio API Error:", errorDetails);
      res.status(error.response?.status || 500).json({ 
        error: "Failed to generate audio from Fish Audio",
        details: errorDetails || error.message 
      });
    }
  });

  // API Route for ElevenLabs Voices Proxy
  app.get("/api/tts/elevenlabs/voices", async (req, res) => {
    const authHeader = req.headers['xi-api-key'] as string;
    const apiKey = authHeader || process.env.ELEVENLABS_API_KEY;

    if (!apiKey) {
      return res.status(400).json({ error: "ElevenLabs API Key is required." });
    }

    try {
      const response = await axios.get("https://api.elevenlabs.io/v1/voices", {
        headers: { 'xi-api-key': apiKey.trim() }
      });
      res.json(response.data);
    } catch (error: any) {
      const status = error.response?.status || 500;
      const errorData = error.response?.data || { error: error.message };
      console.error(`ElevenLabs Voices Error (${status}):`, JSON.stringify(errorData));
      res.status(status).json(errorData);
    }
  });

  // API Route for ElevenLabs TTS Proxy
  app.post("/api/tts/elevenlabs/:voiceId", async (req, res) => {
    const { voiceId } = req.params;
    const { text, model_id, voice_settings } = req.body;
    const authHeader = req.headers['xi-api-key'] as string;
    const apiKey = authHeader || process.env.ELEVENLABS_API_KEY;

    if (!apiKey) {
      return res.status(400).json({ error: "ElevenLabs API Key is required." });
    }

    try {
      const response = await axios.post(
        `https://api.elevenlabs.io/v1/text-to-speech/${voiceId}`,
        { text, model_id, voice_settings },
        {
          headers: {
            'xi-api-key': apiKey.trim(),
            'Content-Type': 'application/json'
          },
          responseType: "arraybuffer",
        }
      );

      res.set("Content-Type", "audio/mpeg");
      res.send(Buffer.from(response.data));
    } catch (error: any) {
      const status = error.response?.status || 500;
      let errorData;
      
      // If the response is an arraybuffer, we need to convert it to a string to see the error message
      if (error.response?.data instanceof ArrayBuffer || Buffer.isBuffer(error.response?.data)) {
        try {
          const decoder = new TextDecoder('utf-8');
          errorData = JSON.parse(decoder.decode(error.response.data));
        } catch (e) {
          errorData = { error: error.message };
        }
      } else {
        errorData = error.response?.data || { error: error.message };
      }

      console.error(`ElevenLabs TTS Error (${status}):`, JSON.stringify(errorData));
      res.status(status).json(errorData);
    }
  });

  // API Route for Groq (OpenAI-compatible Proxy) - used for story/prompt generation
  app.post("/api/groq", async (req, res) => {
    // Soporta hasta 2 API keys de Groq (por si la primera agota su cuota
    // gratuita). GROQ_API_KEY_2 es opcional: si no está configurada, se
    // comporta igual que antes (solo intenta con GROQ_API_KEY).
    const apiKeys = [process.env.GROQ_API_KEY, process.env.GROQ_API_KEY_2].filter(Boolean) as string[];

    if (apiKeys.length === 0) {
      return res.status(500).json({ error: "GROQ_API_KEY is not configured on the server." });
    }

    let lastError: any = null;
    for (let i = 0; i < apiKeys.length; i++) {
      try {
        const response = await axios.post(
          "https://api.groq.com/openai/v1/chat/completions",
          req.body,
          {
            headers: {
              "Content-Type": "application/json",
              "Authorization": `Bearer ${apiKeys[i]}`
            }
          }
        );
        return res.json(response.data);
      } catch (error: any) {
        const status = error.response?.status || 500;
        lastError = error;
        console.error(`Groq API Error con key #${i + 1} (${status}):`, JSON.stringify(error.response?.data || error.message));
        // Solo probamos la siguiente key si esta se quedó sin cuota/límite de velocidad (429).
        // Cualquier otro error (400, 401, etc.) se devuelve de inmediato, sin sentido reintentar con otra key.
        if (status !== 429 || i === apiKeys.length - 1) {
          const errorData = error.response?.data || { error: error.message };
          return res.status(status).json(errorData);
        }
        console.warn(`Key #${i + 1} de Groq alcanzó su límite (429), probando con la siguiente key...`);
      }
    }

    // No debería llegar aquí, pero por si acaso:
    const status = lastError?.response?.status || 500;
    const errorData = lastError?.response?.data || { error: lastError?.message || "Unknown Groq error" };
    res.status(status).json(errorData);
  });

  // API Route for Claude (Anthropic Proxy)
  app.post("/api/claude", async (req, res) => {
    const API_KEY = process.env.ANTHROPIC_API_KEY;
    
    if (!API_KEY) {
      return res.status(500).json({ error: "ANTHROPIC_API_KEY is not configured on the server." });
    }

    try {
      const response = await axios.post(
        "https://api.anthropic.com/v1/messages",
        req.body,
        {
          headers: {
            "Content-Type": "application/json",
            "x-api-key": API_KEY,
            "anthropic-version": "2023-06-01"
          }
        }
      );
      res.json(response.data);
    } catch (error: any) {
      const status = error.response?.status || 500;
      const errorData = error.response?.data || { error: error.message };
      console.error(`Anthropic API Error (${status}):`, JSON.stringify(errorData));
      res.status(status).json(errorData);
    }
  });

  // Health check
  app.get("/api/health", (req, res) => {
    res.json({ status: "ok" });
  });

  // Vite middleware for development
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
    console.log("Vite middleware loaded in development mode");
  } else {
    const distPath = path.join(__dirname, 'dist');
    console.log(`Serving static files from: ${distPath}`);
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      const indexPath = path.join(distPath, 'index.html');
      res.sendFile(indexPath);
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();
