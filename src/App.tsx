/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect, useCallback, useRef } from 'react';
import axios from 'axios';
import { GoogleGenAI, Modality, Type, HarmCategory, HarmBlockThreshold } from "@google/genai";
import { 
  Sparkles, 
  Image as ImageIcon, 
  Video, 
  BookOpen, 
  User, 
  UserPlus,
  Settings, 
  Key,
  Loader2, 
  Download,
  AlertCircle,
  AlertTriangle,
  FileVideo,
  Hash,
  Mic2,
  Play,
  Pause,
  Music,
  Layers,
  Upload,
  X,
  Check,
  CheckCircle2,
  Lock,
  Crown,
  Anchor,
  ExternalLink,
  Zap,
  RefreshCw,
  RotateCw,
  Type as TypeIcon,
  LayoutGrid,
  Copy,
  FileText,
  Clock,
  Volume2,
  MicOff,
  History,
  Info,
  ChevronRight,
  ChevronLeft,
  Share2,
  LogOut,
  Menu,
  Search,
  MoreVertical,
  ShieldCheck,
  ShieldAlert,
  Eye,
  EyeOff
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';
import { motion, AnimatePresence } from 'motion/react';
import JSZip from 'jszip';
import { 
  auth, 
  db, 
  loginWithGoogle, 
  logout, 
  handleFirestoreError, 
  OperationType,
  sanitizeForFirestore
} from './firebase';
import { 
  onAuthStateChanged, 
  User as FirebaseUser 
} from 'firebase/auth';
import { 
  collection, 
  addDoc, 
  updateDoc,
  onSnapshot, 
  query, 
  orderBy, 
  serverTimestamp, 
  doc, 
  setDoc, 
  getDoc,
  deleteDoc,
  where
} from 'firebase/firestore';

// --- Global Types ---
declare global {
  interface Window {
    aistudio?: {
      hasSelectedApiKey: () => Promise<boolean>;
      openSelectKey: () => Promise<void>;
    };
  }
}

// Utility for tailwind classes
function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// Utility to convert raw PCM to WAV
async function createCinematicVideoBlob(imageUrl: string, durationMs: number = 5000, aspectRatio: string = '16:9'): Promise<Blob> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.src = imageUrl;
    
    img.onload = () => {
      const canvas = document.createElement('canvas');
      const [w, h] = aspectRatio === '16:9' ? [1280, 720] : [720, 1280];
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext('2d');
      if (!ctx) return reject(new Error("No se pudo obtener el contexto del canvas"));

      const stream = canvas.captureStream(30); // 30 FPS
      
      const mimeTypes = ['video/webm;codecs=vp9', 'video/webm', 'video/mp4'];
      let selectedMime = mimeTypes.find(m => MediaRecorder.isTypeSupported(m)) || 'video/webm';
      
      const recorder = new MediaRecorder(stream, { mimeType: selectedMime });
      const chunks: Blob[] = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.push(e.data);
      };
      
      recorder.onstop = () => {
        const blob = new Blob(chunks, { type: selectedMime });
        resolve(blob);
      };

      recorder.start();

      const startTime = performance.now();
      const animate = (time: number) => {
        const elapsed = time - startTime;
        const progress = Math.min(elapsed / durationMs, 1);

        const scale = 1 + (progress * 0.15); 
        const xOffset = progress * 30; 
        
        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, w, h);
        
        const imgAspect = img.width / img.height;
        const canvasAspect = w / h;
        let drawW, drawH, drawX, drawY;

        if (imgAspect > canvasAspect) {
          drawH = h;
          drawW = h * imgAspect;
          drawX = (w - drawW) / 2;
          drawY = 0;
        } else {
          drawW = w;
          drawH = w / imgAspect;
          drawX = 0;
          drawY = (h - drawH) / 2;
        }

        ctx.save();
        ctx.translate(w/2, h/2);
        ctx.scale(scale, scale);
        ctx.translate(-w/2, -h/2);
        ctx.drawImage(img, drawX - xOffset, drawY, drawW, drawH);
        ctx.restore();

        if (progress < 1) {
          requestAnimationFrame(animate);
        } else {
          recorder.stop();
        }
      };

      requestAnimationFrame(animate);
    };
    
    img.onerror = () => reject(new Error("No se pudo cargar la imagen para el video"));
  });
}

async function audioBufferToWavBlob(buffer: AudioBuffer): Promise<Blob> {
  const numChannels = buffer.numberOfChannels;
  const sampleRate = buffer.sampleRate;
  const format = 1; // PCM
  const bitDepth = 16;
  
  const bytesPerSample = bitDepth / 8;
  const blockAlign = numChannels * bytesPerSample;
  
  const dataLen = buffer.length * blockAlign;
  const bufferLen = 44 + dataLen;
  const arrayBuffer = new ArrayBuffer(bufferLen);
  const view = new DataView(arrayBuffer);
  
  // RIFF identifier
  view.setUint32(0, 0x52494646, false); // "RIFF"
  view.setUint32(4, 36 + dataLen, true);
  view.setUint32(8, 0x57415645, false); // "WAVE"
  
  // fmt chunk
  view.setUint32(12, 0x666d7420, false); // "fmt "
  view.setUint32(16, 16, true);
  view.setUint16(20, format, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitDepth, true);
  
  // data chunk
  view.setUint32(36, 0x64617461, false); // "data"
  view.setUint32(40, dataLen, true);
  
  // Write samples
  const offset = 44;
  if (numChannels === 2) {
    const left = buffer.getChannelData(0);
    const right = buffer.getChannelData(1);
    for (let i = 0; i < buffer.length; i++) {
      let l = Math.max(-1, Math.min(1, left[i]));
      let r = Math.max(-1, Math.min(1, right[i]));
      view.setInt16(offset + i * 4, l < 0 ? l * 0x8000 : l * 0x7FFF, true);
      view.setInt16(offset + i * 4 + 2, r < 0 ? r * 0x8000 : r * 0x7FFF, true);
    }
  } else {
    const channel = buffer.getChannelData(0);
    for (let i = 0; i < buffer.length; i++) {
      let s = Math.max(-1, Math.min(1, channel[i]));
      view.setInt16(offset + i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
  }
  
  return new Blob([arrayBuffer], { type: 'audio/wav' });
}

function pcmToWav(pcmBase64: string, sampleRate: number = 24000): { url: string; blob: Blob } {
  const binaryString = atob(pcmBase64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }

  const wavHeader = new ArrayBuffer(44);
  const view = new DataView(wavHeader);

  // RIFF identifier
  view.setUint32(0, 0x52494646, false); // "RIFF"
  // file length
  view.setUint32(4, 36 + len, true);
  // RIFF type
  view.setUint32(8, 0x57415645, false); // "WAVE"
  // format chunk identifier
  view.setUint32(12, 0x666d7420, false); // "fmt "
  // format chunk length
  view.setUint32(16, 16, true);
  // sample format (raw PCM = 1)
  view.setUint16(20, 1, true);
  // channel count
  view.setUint16(22, 1, true);
  // sample rate
  view.setUint32(24, sampleRate, true);
  // byte rate (sample rate * block align)
  view.setUint32(28, sampleRate * 2, true);
  // block align (channel count * bytes per sample)
  view.setUint16(32, 2, true);
  // bits per sample
  view.setUint16(34, 16, true);
  // data chunk identifier
  view.setUint32(36, 0x64617461, false); // "data"
  // data chunk length
  view.setUint32(40, len, true);

  const blob = new Blob([wavHeader, bytes], { type: 'audio/wav' });
  return { url: URL.createObjectURL(blob), blob };
}

// --- Groq (motor de texto/visión: historias, guiones, prompts y análisis de referencias) ---

interface GroqContentPart {
  type: "text" | "image_url";
  text?: string;
  image_url?: { url: string };
}

interface GroqChatMessage {
  role: "system" | "user" | "assistant";
  content: string | GroqContentPart[];
}

async function callGroq(messages: GroqChatMessage[], model: string = "openai/gpt-oss-120b"): Promise<string> {
  const body: any = {
    model,
    messages,
    temperature: 0.9,
    max_completion_tokens: 16000,
    response_format: { type: "json_object" }
  };
  // reasoning_effort solo aplica a los modelos de razonamiento (openai/gpt-oss-*),
  // no a los modelos de visión (qwen3.x), que no soportan este parámetro.
  if (model.startsWith("openai/gpt-oss")) {
    body.reasoning_effort = "low";
  }

  const response = await fetch("/api/groq", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });

  if (!response.ok) {
    let details = "";
    try {
      const errJson = await response.json();
      details = errJson.error?.message || errJson.error || JSON.stringify(errJson);
    } catch {
      details = await response.text();
    }
    throw new Error(`Groq API error (${response.status}): ${details}`);
  }

  const data = await response.json();
  const content = data.choices?.[0]?.message?.content;
  const finishReason = data.choices?.[0]?.finish_reason;
  if (!content) throw new Error("Groq no devolvió contenido en la respuesta.");
  if (finishReason === "length") {
    throw new Error("La respuesta de Groq se cortó por exceder el límite de tokens (JSON incompleto). Prueba con una duración/número de segmentos menor.");
  }
  return content;
}

// Groq (a diferencia de Gemini) no fuerza un schema estricto, solo JSON válido.
// Esta función limpia envolturas de markdown y extrae el primer objeto JSON completo.
function parseGroqJson(rawText: string): any {
  let cleaned = rawText.trim();
  cleaned = cleaned.replace(/^```(?:json)?\s*/i, "").replace(/```\s*$/i, "").trim();

  try {
    return JSON.parse(cleaned);
  } catch {
    const start = cleaned.indexOf('{');
    const end = cleaned.lastIndexOf('}');
    if (start !== -1 && end !== -1 && end > start) {
      return JSON.parse(cleaned.slice(start, end + 1));
    }
    throw new Error("No se pudo interpretar la respuesta JSON de Groq.");
  }
}

// --- Types ---

interface StorySegment {
  id: string;
  text: string;
  imagePrompt: string;
  videoDescription?: string;
  cameraAngle?: string;
  imageUrl?: string;
  videoUrl?: string;
  audioUrl?: string;
  audioStart?: number;
  audioEnd?: number;
  duration?: number; // Duración estimada o real en segundos
  characterVoice?: string;
  videoBlob?: Blob;
  audioBlob?: Blob;
  isGeneratingImage?: boolean;
  isGeneratingVideo?: boolean;
  isGeneratingAudio?: boolean; // Added for per-segment audio generation
  error?: string;
}

interface CharacterProfile {
  id: string;
  name: string;
  description: string;
  style: VisualStyle;
  createdAt: number;
}

interface Story {
  id?: string;
  title: string;
  narration: string;
  mainCharacterDescription?: string;
  hashtags: string[];
  segments: StorySegment[];
  audioUrl?: string;
  audioBlob?: Blob;
  backgroundMusicUrl?: string;
  backgroundMusicBlob?: Blob;
  createdAt?: number;
  type: 'quote' | 'narrative' | 'analyze';
  subtitleStyle?: 'classic' | 'minimal' | 'bold' | 'cinematic' | 'none';
  subtitlePosition?: 'top' | 'middle' | 'bottom';
  voiceProvider?: 'gemini' | 'elevenlabs';
  elevenLabsVoiceId?: string;
  isCustomAudio?: boolean;
}

type ImageSize = "1K" | "2K" | "4K";
type VideoAspectRatio = "16:9" | "9:16" | "1:1" | "4:3";
const ALL_STYLES = [
  "Cinematic", "Anime", "Oil Painting", "Comic Book", "Pixar Anime", "3D Animation", "Reflection Dark", "Reflection Nature", "Reflection Urban", "Gold Skeleton", "Glass Skeleton", "Cyber-Schematic", "Manhwa Premium", "Neon Blueprint", "Neon Pro", "HOLOGRAPHIC SOUL", "Power Couple", "Comic Realista Moderno", "Dios", "Realismo Crudo", "Minimal Clay Animation", "Muscular Fitness", "Historias de Dios"
] as const;

type VisualStyle = typeof ALL_STYLES[number];

const VISUAL_STYLES = [
  "Neon Pro",
  "HOLOGRAPHIC SOUL",
  "3D Animation",
  "Comic Realista Moderno",
  "Minimal Clay Animation",
  "Muscular Fitness",
  "Historias de Dios"
] as const;
type CameraAngle = "Standard" | "Close-up" | "Extreme Close-up" | "Conversation" | "Aerial" | "Wide Shot" | "Fisheye" | "Low Angle" | "Dutch Angle" | "Mirror Reflection";
type StoryDuration = "Auto" | "30s" | "1m" | "2m" | "3m" | "10imgs" | "20imgs" | "30imgs" | "40imgs" | "50imgs" | "100imgs";
type ModelQuality = "Pro" | "Savings";
type AppMode = "create";

// --- Components ---

const FinalPreviewPlayer = ({ 
  story, 
  onClose, 
  backgroundMusicMood,
  videoAspectRatio,
  visualStyle,
  onDownload,
  onExport,
  videoProvider,
  subtitleStyle = 'classic',
  subtitlePosition = 'bottom'
}: { 
  story: Story, 
  onClose: () => void,
  backgroundMusicMood: string,
  videoAspectRatio: VideoAspectRatio,
  visualStyle: VisualStyle,
  onDownload: () => void,
  onExport: () => void,
  videoProvider: "Veo" | "CinematicPan",
  subtitleStyle?: 'classic' | 'minimal' | 'bold' | 'cinematic' | 'divine' | 'none',
  subtitlePosition?: 'top' | 'middle' | 'bottom'
}) => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(true);
  const [audioError, setAudioError] = useState<string | null>(null);
  const videoRef = React.useRef<HTMLVideoElement>(null);
  const audioRef = React.useRef<HTMLAudioElement>(null);
  const bgMusicRef = React.useRef<HTMLAudioElement>(null);

  // Calcular tiempos de inicio de cada segmento para sincronizar con el audio global
  const segmentStartTimes = React.useMemo(() => {
    let current = 0;
    return story.segments.map(s => {
      const start = current;
      current += s.duration || 5;
      return start;
    });
  }, [story.segments]);

  const currentSegment = story.segments[currentIndex];
  const isCinematicPan = videoProvider === "CinematicPan" && currentSegment?.imageUrl;

  const handleNext = useCallback(() => {
    if (currentIndex < story.segments.length - 1) {
      const nextIdx = currentIndex + 1;
      setCurrentIndex(nextIdx);
      if (story.audioUrl && audioRef.current) {
        audioRef.current.currentTime = segmentStartTimes[nextIdx];
      }
    } else {
      setIsPlaying(false);
    }
  }, [currentIndex, story.segments.length, story.audioUrl, segmentStartTimes]);

  const handlePrev = useCallback(() => {
    if (currentIndex > 0) {
      const prevIdx = currentIndex - 1;
      setCurrentIndex(prevIdx);
      if (story.audioUrl && audioRef.current) {
        audioRef.current.currentTime = segmentStartTimes[prevIdx];
      }
    }
  }, [currentIndex, story.audioUrl, segmentStartTimes]);

  const handleTimeUpdate = (e: React.SyntheticEvent<HTMLAudioElement>) => {
    if (!story.audioUrl) return;
    const currentTime = e.currentTarget.currentTime;
    
    // Encontrar el segmento que corresponde al tiempo actual de forma más precisa
    let nextIndex = -1;
    for (let i = segmentStartTimes.length - 1; i >= 0; i--) {
      if (currentTime >= segmentStartTimes[i] - 0.1) { // Pequeño margen de 100ms para suavidad
        nextIndex = i;
        break;
      }
    }

    if (nextIndex !== -1 && nextIndex !== currentIndex) {
      setCurrentIndex(nextIndex);
    }
  };

  useEffect(() => {
    if (isPlaying && videoRef.current && !isCinematicPan) {
      videoRef.current.play().catch(err => console.error("Video play error:", err));
    } else if (!isPlaying && videoRef.current) {
      videoRef.current.pause();
    }

    if (isPlaying && audioRef.current) {
      audioRef.current.play().catch(err => {
        console.error("Audio play error:", err);
        setAudioError("El audio no pudo reproducirse automáticamente. Haz clic en el botón de audio.");
      });
    } else if (!isPlaying && audioRef.current) {
      audioRef.current.pause();
    }

    if (isPlaying && bgMusicRef.current) {
      bgMusicRef.current.volume = 0.15;
      bgMusicRef.current.play().catch(err => console.error("BG Music play error:", err));
    } else if (!isPlaying && bgMusicRef.current) {
      bgMusicRef.current.pause();
    }

    // Auto-advance if it's an image or CinematicPan AND there's no audio (global or segment)
    if (isPlaying && (!currentSegment?.videoUrl || isCinematicPan) && !currentSegment?.audioUrl && !story.audioUrl) {
      const timer = setTimeout(() => {
        handleNext();
      }, visualStyle === "Dios" ? 7000 : 5000); // 5 seconds per image if no audio (7s for Dios style)
      return () => clearTimeout(timer);
    }
  }, [currentIndex, isPlaying, isCinematicPan, currentSegment, story.audioUrl, handleNext]);

  const getMusicUrl = (mood: string) => {
    switch (mood) {
      case "Epic": return "https://cdn.pixabay.com/download/audio/2022/03/10/audio_c8c8a73456.mp3"; // Epic
      case "Emotional": return "https://cdn.pixabay.com/download/audio/2022/01/26/audio_d0c6ff1101.mp3"; // Emotional
      case "Cyberpunk": return "https://cdn.pixabay.com/download/audio/2022/03/15/audio_18c4a16677.mp3"; // Cyberpunk
      case "Nature": return "https://cdn.pixabay.com/download/audio/2022/02/22/audio_d1c6ff1101.mp3"; // Nature
      case "Horror": return "https://cdn.pixabay.com/download/audio/2022/03/10/audio_c8c8a73456.mp3"; // Placeholder
      default: return null;
    }
  };

  return (
    <motion.div 
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-[100] bg-black flex flex-col items-center justify-center p-4 sm:p-8"
    >
      <div className="absolute top-6 right-6 flex gap-4 z-10">
        <button 
          onClick={onExport}
          className="p-3 bg-emerald-500 hover:bg-emerald-400 rounded-full backdrop-blur-md transition-all border border-emerald-400/20 shadow-lg shadow-emerald-500/20 flex items-center gap-2 px-5"
          title="Exportar como Video MP4"
        >
          <Video className="w-5 h-5 text-black" />
          <span className="text-xs font-bold text-black uppercase tracking-wider">Exportar MP4</span>
        </button>
        <button 
          onClick={onDownload}
          className="p-3 bg-white/10 hover:bg-white/20 rounded-full backdrop-blur-md transition-all border border-white/10"
          title="Descargar ZIP de Recursos"
        >
          <Download className="w-6 h-6 text-white" />
        </button>
        <button 
          onClick={onClose}
          className="p-3 bg-white/10 hover:bg-white/20 rounded-full backdrop-blur-md transition-all border border-white/10"
        >
          <X className="w-6 h-6 text-white" />
        </button>
      </div>

      <div className={cn(
        "relative w-full bg-zinc-900 rounded-3xl overflow-hidden shadow-2xl border border-white/5",
        videoAspectRatio === "16:9" ? "max-w-4xl aspect-video" : 
        videoAspectRatio === "9:16" ? "max-w-[380px] aspect-[9/16]" :
        videoAspectRatio === "1:1" ? "max-w-2xl aspect-square" : "max-w-4xl aspect-[4/3]"
      )}>
        {currentSegment?.videoUrl && !isCinematicPan ? (
          <video 
            ref={videoRef}
            src={currentSegment.videoUrl} 
            onEnded={(!currentSegment.audioUrl && !story.audioUrl) ? handleNext : undefined}
            className="w-full h-full object-cover"
            autoPlay={isPlaying}
          />
        ) : isCinematicPan ? (
          <img 
            src={currentSegment.imageUrl} 
            className={cn(
              "w-full h-full object-cover",
              isPlaying ? "animate-cinematic-pan" : ""
            )}
            style={{ 
              animationDuration: `${(currentSegment.duration || 5) * 2}s` // *2 because it's a yoyo animation (0-50-100)
            }}
            alt="Cinematic Pan"
            referrerPolicy="no-referrer"
          />
        ) : currentSegment?.imageUrl ? (
          <img 
            src={currentSegment.imageUrl} 
            className="w-full h-full object-cover"
            alt="Segment visual"
            referrerPolicy="no-referrer"
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center bg-zinc-900">
            <Loader2 className="w-12 h-12 text-emerald-500 animate-spin" />
          </div>
        )}

        {/* Subtitles Overlay */}
        {subtitleStyle !== 'none' && (
          <div className={cn(
            "absolute left-0 right-0 flex justify-center px-8 pointer-events-none z-20 transition-all duration-500",
            subtitlePosition === 'top' ? "top-12" : 
            subtitlePosition === 'middle' ? "top-1/2 -translate-y-1/2" : "bottom-12"
          )}>
            <div className={cn(
              "text-center transition-all duration-300",
              subtitleStyle === 'classic' && "bg-black/60 backdrop-blur-md px-6 py-3 rounded-2xl border border-white/10 max-w-2xl shadow-2xl text-white text-lg sm:text-xl font-bold leading-tight tracking-tight",
              subtitleStyle === 'minimal' && "max-w-2xl text-white text-lg sm:text-xl font-bold leading-tight tracking-tight [text-shadow:_0_2px_4px_rgb(0_0_0_/_80%)]",
              subtitleStyle === 'bold' && "bg-yellow-400 px-6 py-2 rounded-lg max-w-2xl shadow-xl text-black text-xl sm:text-2xl font-black uppercase italic leading-none tracking-tighter transform -rotate-1",
              subtitleStyle === 'cinematic' && "max-w-3xl text-zinc-100 text-base sm:text-lg font-serif italic leading-relaxed tracking-wide opacity-90 [text-shadow:_0_1px_2px_rgb(0_0_0_/_50%)]",
              subtitleStyle === 'divine' && "max-w-2xl text-white text-2xl sm:text-4xl font-black uppercase tracking-tighter [text-shadow:_0_4px_8px_rgba(0,0,0,0.9),_0_0_20px_rgba(255,255,255,0.2)] leading-none",
            )}>
              <p>{currentSegment?.text}</p>
            </div>
          </div>
        )}

        {(!currentSegment?.audioUrl && !story.audioUrl) && isPlaying && (
          <div className="absolute top-4 left-4 bg-amber-500/90 text-black text-[9px] font-black px-3 py-1.5 rounded-xl uppercase tracking-widest z-30 flex items-center gap-2 shadow-xl border border-amber-400/20">
            <MicOff className="w-3 h-3" />
            Escena sin audio
          </div>
        )}

        {/* Progress Bar */}
        <div className="absolute bottom-0 left-0 right-0 h-1.5 bg-white/10">
          <motion.div 
            className="h-full bg-emerald-500"
            initial={{ width: "0%" }}
            animate={{ width: `${((currentIndex + 1) / story.segments.length) * 100}%` }}
            transition={{ duration: 0.5 }}
          />
        </div>
      </div>

      {/* Controls */}
      <div className="mt-8 flex items-center gap-6">
        <button 
          onClick={handlePrev}
          disabled={currentIndex === 0}
          className="p-4 bg-white/10 hover:bg-white/20 rounded-2xl disabled:opacity-30 transition-all border border-white/10"
        >
          <Play className="w-6 h-6 text-white rotate-180" />
        </button>

        <button 
          onClick={() => setIsPlaying(!isPlaying)}
          className="p-6 bg-emerald-500 hover:bg-emerald-400 rounded-3xl transition-all shadow-xl shadow-emerald-500/20"
        >
          {isPlaying ? <Pause className="w-8 h-8 text-black" /> : <Play className="w-8 h-8 text-black" />}
        </button>

        <button 
          onClick={handleNext}
          disabled={currentIndex === story.segments.length - 1}
          className="p-4 bg-white/10 hover:bg-white/20 rounded-2xl disabled:opacity-30 transition-all border border-white/10"
        >
          <Play className="w-6 h-6 text-white" />
        </button>

        {(currentSegment?.audioUrl || story.audioUrl) && (
          <button 
            onClick={() => {
              if (audioRef.current) {
                if (story.audioUrl) {
                  audioRef.current.currentTime = segmentStartTimes[currentIndex];
                } else {
                  audioRef.current.currentTime = 0;
                }
                audioRef.current.play();
                setIsPlaying(true);
              }
            }}
            className="p-4 bg-emerald-500/20 hover:bg-emerald-500/30 rounded-2xl transition-all border border-emerald-500/30"
            title="Reiniciar Escena"
          >
            <Mic2 className="w-6 h-6 text-emerald-400" />
          </button>
        )}
      </div>

      {audioError && (
        <div className="mt-4 px-4 py-2 bg-amber-500/20 border border-amber-500/30 rounded-xl text-amber-400 text-[10px] font-bold uppercase tracking-widest">
          {audioError}
        </div>
      )}

      <div className="mt-6 text-zinc-500 font-bold uppercase tracking-widest text-xs">
        Escena {currentIndex + 1} de {story.segments.length}
      </div>

      {/* Hidden Audio Elements */}
      {(currentSegment?.audioUrl || story.audioUrl) && (
        <audio 
          ref={audioRef} 
          src={story.audioUrl || currentSegment.audioUrl} 
          onEnded={handleNext}
          onTimeUpdate={handleTimeUpdate}
          autoPlay={isPlaying}
        />
      )}
      {backgroundMusicMood !== "None" && (
        <audio 
          ref={bgMusicRef} 
          src={getMusicUrl(backgroundMusicMood) || ""} 
          loop 
          autoPlay={isPlaying}
        />
      )}
    </motion.div>
  );
};

const SegmentDisplay = ({ 
  segment, 
  index, 
  storyDuration, 
  visualStyle, 
  videoAspectRatio,
  generateImage,
  generateVideo,
  story
}: { 
  segment: StorySegment, 
  index: number,
  storyDuration: string,
  visualStyle: string,
  videoAspectRatio: string,
  generateImage: (i: number) => void,
  generateVideo: (i: number) => void,
  story: Story | null
}) => {
  return (
    <motion.div
      key={segment.id}
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -20 }}
      className="space-y-4"
    >
      {/* Visual Display */}
      <div className={cn(
        "relative bg-zinc-900 rounded-2xl overflow-hidden border border-zinc-800 group transition-all duration-500",
        videoAspectRatio === "16:9" ? "aspect-video" : 
        videoAspectRatio === "9:16" ? "aspect-[9/16] max-w-[320px] mx-auto" :
        videoAspectRatio === "1:1" ? "aspect-square max-w-[400px] mx-auto" :
        "aspect-[4/3] max-w-[500px] mx-auto"
      )}>
        {segment.videoUrl ? (
          <div className="relative w-full h-full">
            <video src={segment.videoUrl} controls autoPlay loop className="w-full h-full object-cover" />
          </div>
        ) : segment.imageUrl ? (
          <div className="relative w-full h-full">
            <img src={segment.imageUrl} alt="Segment visual" className="w-full h-full object-cover" />
          </div>
        ) : (
          <div className="absolute inset-0 flex flex-col items-center justify-center space-y-4">
            {segment.isGeneratingImage || segment.isGeneratingVideo ? (
              <div className="flex flex-col items-center gap-4">
                <div className="relative">
                  <Loader2 className="w-12 h-12 text-emerald-500 animate-spin" />
                  <div className="absolute inset-0 blur-xl bg-emerald-500/20 animate-pulse" />
                </div>
                <span className="text-sm font-medium text-zinc-400 animate-pulse">
                  {segment.isGeneratingVideo 
                    ? (segment.imageUrl ? "Animando Imagen Pro (Veo 3.1)..." : "Sintetizando Video Pro (Veo 3.1 HQ)...") 
                    : "Visualizando Escena..."}
                </span>
              </div>
            ) : (
              <>
                <ImageIcon className="w-12 h-12 text-zinc-800" />
                <span className="text-sm text-zinc-600">Visual aún no generada</span>
              </>
            )}
          </div>
        )}

        {/* Action Overlay */}
        <div className="absolute bottom-4 right-4 flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
          <button 
            onClick={() => generateImage(index)}
            disabled={segment.isGeneratingImage}
            className="p-2 bg-black/80 backdrop-blur-md border border-white/10 rounded-lg hover:bg-white hover:text-black transition-all disabled:opacity-50"
            title="Generar Imagen"
          >
            <ImageIcon className="w-4 h-4" />
          </button>
          <button 
            onClick={() => generateVideo(index)}
            disabled={segment.isGeneratingVideo}
            className="p-2 bg-black/80 backdrop-blur-md border border-white/10 rounded-lg hover:bg-white hover:text-black transition-all disabled:opacity-50"
            title="Animar"
          >
            <Video className="w-4 h-4" />
          </button>
          {(segment.videoUrl || segment.imageUrl) && (
            <button 
              onClick={async () => {
                const url = segment.videoUrl || segment.imageUrl;
                if (!url) return;
                
                try {
                  if (url.startsWith('data:')) {
                    const link = document.createElement('a');
                    link.href = url;
                    link.download = segment.videoUrl ? `segment_${index + 1}_video.mp4` : `segment_${index + 1}_image.png`;
                    link.click();
                    return;
                  }

                  const response = await fetch(url);
                  const blob = await response.blob();
                  const blobUrl = URL.createObjectURL(blob);
                  const link = document.createElement('a');
                  link.href = blobUrl;
                  link.download = segment.videoUrl ? `segment_${index + 1}_video.mp4` : `segment_${index + 1}_image.png`;
                  link.click();
                  URL.revokeObjectURL(blobUrl);
                } catch (err) {
                  const link = document.createElement('a');
                  link.href = url;
                  link.download = segment.videoUrl ? `segment_${index + 1}_video.mp4` : `segment_${index + 1}_image.png`;
                  link.click();
                }
              }}
              className="p-2 bg-emerald-500/80 backdrop-blur-md border border-emerald-400/20 rounded-lg hover:bg-emerald-400 text-black transition-all"
              title="Descargar"
            >
              <Download className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>

      {/* Text Content */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="md:col-span-2 space-y-3">
          <div className="flex items-center gap-2">
            <span className="px-2 py-0.5 bg-emerald-500/10 text-emerald-500 text-[9px] font-bold rounded uppercase tracking-wider">
              Segmento {index + 1}
            </span>
            {segment.cameraAngle && (
              <span className="px-2 py-0.5 bg-zinc-800 text-zinc-400 text-[9px] font-bold rounded uppercase tracking-wider flex items-center gap-1">
                <Video className="w-2.5 h-2.5" />
                {segment.cameraAngle}
              </span>
            )}
            <div className="h-px flex-1 bg-zinc-800" />
            {segment.audioUrl ? (
              <span className="px-2 py-0.5 bg-emerald-500/10 text-emerald-500 text-[9px] font-bold rounded uppercase tracking-wider flex items-center gap-1">
                <Volume2 className="w-2.5 h-2.5" />
                Audio Listo
              </span>
            ) : story?.audioUrl ? (
              <span className="px-2 py-0.5 bg-blue-500/10 text-blue-500 text-[9px] font-bold rounded uppercase tracking-wider flex items-center gap-1">
                <Volume2 className="w-2.5 h-2.5" />
                Audio Global
              </span>
            ) : (
              <span className="px-2 py-0.5 bg-amber-500/10 text-amber-500 text-[9px] font-bold rounded uppercase tracking-wider flex items-center gap-1">
                <MicOff className="w-2.5 h-2.5" />
                Sin Audio
              </span>
            )}
          </div>
          <div className="prose prose-invert max-w-none text-base leading-relaxed text-zinc-300">
            <ReactMarkdown>{segment.text}</ReactMarkdown>
          </div>
        </div>

        <div className="space-y-3">
          <div className="p-3 bg-zinc-900/30 border border-zinc-800 rounded-xl space-y-2">
            <h4 className="text-[9px] font-bold uppercase tracking-wider text-zinc-500">Prompt Visual</h4>
            <p className="text-[11px] text-zinc-400 leading-relaxed italic">"{segment.imagePrompt}"</p>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

const QUICK_THEMES = [
  { 
    name: "Estilo Neón", 
    prompt: "Mensajes inspiradores con estética futurista de neón.", 
    visualStyle: "Neon Pro" as VisualStyle 
  },
  { 
    name: "Animación 3D", 
    prompt: "Reflexiones profundas sobre la vida con estética de animación 3D.", 
    visualStyle: "3D Animation" as VisualStyle 
  },
  { 
    name: "Cómic Realista", 
    prompt: "Mensajes motivacionales con estilo de cómic realista.", 
    visualStyle: "Comic Realista Moderno" as VisualStyle 
  },
  { 
    name: "Minimal Clay", 
    prompt: "Mensajes de calma y bienestar con estética minimalista de arcilla.", 
    visualStyle: "Minimal Clay Animation" as VisualStyle 
  },
  { 
    name: "Muscular Fitness", 
    prompt: "Anime/manga illustration style, extremely muscular bald male bodybuilder character, highly detailed muscle anatomy, ultra defined pectorals, abs, quads and arms, shiny sweaty skin texture, wearing black gym shorts, dark professional gym background, dramatic cinematic lighting with deep red/orange accent lights, atmospheric smoke, high contrast shadows.", 
    visualStyle: "Muscular Fitness" as VisualStyle 
  }
];

export default function App() {
  const [user, setUser] = useState<FirebaseUser | null>(null);
  const [isAuthReady, setIsAuthReady] = useState(false);
  const [isAccessDenied, setIsAccessDenied] = useState(false);
  const [userStories, setUserStories] = useState<(Story & { id: string })[]>([]);
  const [currentStoryId, setCurrentStoryId] = useState<string | null>(null);
  const [showMyStories, setShowMyStories] = useState(false);

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (u) => {
      if (u && u.email?.toLowerCase() !== 'brayansulsona14@gmail.com') {
        setIsAccessDenied(true);
        logout();
        setUser(null);
        setIsAuthReady(true);
        return;
      }
      
      setUser(u);
      setIsAccessDenied(false);
      setIsAuthReady(true);
      if (u) {
        // Create or update user profile safely without overwriting createdAt
        const userRef = doc(db, 'users', u.uid);
        const defaultRole = u.email?.toLowerCase() === 'brayansulsona14@gmail.com' ? 'admin' : 'user';
        getDoc(userRef).then(snap => {
          if (!snap.exists()) {
            setDoc(userRef, sanitizeForFirestore({
              uid: u.uid,
              email: u.email || '',
              displayName: u.displayName || '',
              photoURL: u.photoURL || '',
              role: defaultRole,
              createdAt: Date.now()
            })).catch(err => {
              console.error("Error creating user profile:", err);
            });
          } else {
            setDoc(userRef, sanitizeForFirestore({
              email: u.email || '',
              displayName: u.displayName || '',
              photoURL: u.photoURL || '',
              updatedAt: Date.now()
            }), { merge: true }).catch(err => {
              console.error("Error updating user profile:", err);
            });
          }
        }).catch(err => {
          console.error("Error fetching user profile:", err);
        });
      }
    });

    // Safety timeout: if auth doesn't respond in 10s, mark as ready anyway
    // to show the login screen or error state instead of a permanent loader.
    const safetyTimeout = setTimeout(() => {
      setIsAuthReady(prev => {
        if (!prev) {
          console.warn("Auth state check timed out. Proceeding anyway.");
          return true;
        }
        return prev;
      });
    }, 10000);

    return () => {
      unsubscribe();
      clearTimeout(safetyTimeout);
    };
  }, []);

  useEffect(() => {
    if (!user) {
      setUserStories([]);
      return;
    }

    const storiesRef = collection(db, 'users', user.uid, 'stories');
    const q = query(storiesRef, orderBy('createdAt', 'desc'));

    const unsubscribe = onSnapshot(q, (snapshot) => {
      const stories = snapshot.docs.map(doc => ({
        id: doc.id,
        ...doc.data()
      })) as (Story & { id: string })[];
      setUserStories(stories);
    }, (err) => {
      handleFirestoreError(err, OperationType.LIST, `users/${user.uid}/stories`);
    });

    return () => unsubscribe();
  }, [user]);

  const [isSettingsLoaded, setIsSettingsLoaded] = useState(false);

  const [isAuthorized, setIsAuthorized] = useState<boolean>(() => localStorage.getItem('is_authorized') === 'true');
  const [passwordInput, setPasswordInput] = useState<string>('');
  const [passwordError, setPasswordError] = useState<boolean>(false);
  const [apiKeySelected, setApiKeySelected] = useState<boolean>(() => localStorage.getItem('api_key_selected') === 'true');
  const [isAiStudioReady, setIsAiStudioReady] = useState(false);
  const [manualApiKey, setManualApiKey] = useState<string>(() => localStorage.getItem('manual_gemini_api_key') || '');
  const [showApiKeyInput, setShowApiKeyInput] = useState<boolean>(false);
  const [appMode, setAppMode] = useState<AppMode>("create");
  const [sidebarTab, setSidebarTab] = useState<'narrative' | 'quote' | 'analyze'>(() => (localStorage.getItem('app_sidebar_tab') as any) || 'narrative');
  const [analysisResult, setAnalysisResult] = useState<{
    summary: string;
    style: string;
    tone: string;
    prompt: string;
  } | null>(null);
  const [isAnalyzingVideo, setIsAnalyzingVideo] = useState(false);
  const [prompt, setPrompt] = useState(() => localStorage.getItem('app_prompt') || '');
  const [characterDesc, setCharacterDesc] = useState(() => localStorage.getItem('app_character_desc') || '');
  const [imageSize, setImageSize] = useState<ImageSize>(() => (localStorage.getItem('app_image_size') as ImageSize) || "1K");
  const [videoAspectRatio, setVideoAspectRatio] = useState<VideoAspectRatio>(() => (localStorage.getItem('app_video_aspect_ratio') as VideoAspectRatio) || "16:9");
  const [visualStyle, setVisualStyle] = useState<VisualStyle>(() => (localStorage.getItem('app_visual_style') as VisualStyle) || "Neon Pro");
  const [dynamicAngles, setDynamicAngles] = useState(() => localStorage.getItem('app_dynamic_angles') !== 'false');
  const [useExactText, setUseExactText] = useState(() => localStorage.getItem('app_use_exact_text') === 'true');
  const [storyDuration, setStoryDuration] = useState<StoryDuration>(() => (localStorage.getItem('app_story_duration') as StoryDuration) || "Auto");
  const [modelQuality, setModelQuality] = useState<ModelQuality>(() => (localStorage.getItem('app_model_quality') as ModelQuality) || "Pro");
  const [selectedVoice, setSelectedVoice] = useState<string>(() => localStorage.getItem('app_selected_voice') || "Charon");
  const [voiceProvider, setVoiceProvider] = useState<'gemini' | 'elevenlabs'>(() => (localStorage.getItem('app_voice_provider') as any) || 'gemini');
  const [elevenLabsApiKey, setElevenLabsApiKey] = useState<string>(() => localStorage.getItem('app_elevenlabs_api_key') || '');
  const [elevenLabsVoices, setElevenLabsVoices] = useState<{id: string, name: string}[]>(() => {
    try {
      const saved = localStorage.getItem('app_elevenlabs_voices');
      return saved ? JSON.parse(saved) : [];
    } catch (e) {
      return [];
    }
  });
  const [globalAudioOffset, setGlobalAudioOffset] = useState<number>(0);
  const [isSyncMode, setIsSyncMode] = useState(false);
  const [isFetchingVoices, setIsFetchingVoices] = useState(false);
  const [selectedElevenLabsVoice, setSelectedElevenLabsVoice] = useState<string>(() => localStorage.getItem('app_selected_elevenlabs_voice') || '');

  useEffect(() => {
    localStorage.setItem('app_selected_voice', selectedVoice);
  }, [selectedVoice]);

  useEffect(() => {
    localStorage.setItem('app_voice_provider', voiceProvider);
  }, [voiceProvider]);

  useEffect(() => {
    localStorage.setItem('app_selected_elevenlabs_voice', selectedElevenLabsVoice);
  }, [selectedElevenLabsVoice]);

  useEffect(() => {
    localStorage.setItem('app_elevenlabs_api_key', elevenLabsApiKey);
  }, [elevenLabsApiKey]);

  useEffect(() => {
    localStorage.setItem('app_elevenlabs_voices', JSON.stringify(elevenLabsVoices));
  }, [elevenLabsVoices]);

  useEffect(() => {
    localStorage.setItem('app_global_audio_offset', globalAudioOffset.toString());
  }, [globalAudioOffset]);

  useEffect(() => {
    const savedOffset = localStorage.getItem('app_global_audio_offset');
    if (savedOffset) setGlobalAudioOffset(parseFloat(savedOffset));
  }, []);

  useEffect(() => {
    localStorage.setItem('app_global_audio_offset', globalAudioOffset.toString());
  }, [globalAudioOffset]);

  useEffect(() => {
    const savedOffset = localStorage.getItem('app_global_audio_offset');
    if (savedOffset) setGlobalAudioOffset(parseFloat(savedOffset));
  }, []);

  useEffect(() => {
    localStorage.setItem('app_selected_elevenlabs_voice', selectedElevenLabsVoice);
  }, [selectedElevenLabsVoice]);

  const [imageProvider, setImageProvider] = useState<"Google" | "Pollinations">(() => (localStorage.getItem('app_image_provider') as any) || "Pollinations");
  const [videoProvider, setVideoProvider] = useState<"Veo" | "CinematicPan">(() => (localStorage.getItem('app_video_provider') as any) || "CinematicPan");

  useEffect(() => {
    localStorage.setItem('app_image_provider', imageProvider);
  }, [imageProvider]);

  useEffect(() => {
    localStorage.setItem('app_video_provider', videoProvider);
  }, [videoProvider]);
  const [burnTextIntoImages, setBurnTextIntoImages] = useState(() => {
    const saved = localStorage.getItem('app_burn_text');
    return saved === null ? true : saved === 'true';
  });

  const [isGeneratingStory, setIsGeneratingStory] = useState(false);
  const [isGeneratingAudio, setIsGeneratingAudio] = useState(false);
  const [audioProgress, setAudioProgress] = useState<{ current: number; total: number } | null>(null);
  const [isSearchingMusic, setIsSearchingMusic] = useState(false);
  const [isPreviewingVoice, setIsPreviewingVoice] = useState(false);
  const [isBulkGeneratingImages, setIsBulkGeneratingImages] = useState(false);
  const [isBulkGeneratingVideos, setIsBulkGeneratingVideos] = useState(false);
  const [backgroundMusicMood, setBackgroundMusicMood] = useState<string>("None");
  const [subtitleStyle, setSubtitleStyle] = useState<'classic' | 'minimal' | 'bold' | 'cinematic' | 'divine' | 'none'>(() => (localStorage.getItem('app_subtitle_style') as any) || 'divine');
  const [subtitlePosition, setSubtitlePosition] = useState<'top' | 'middle' | 'bottom'>(() => (localStorage.getItem('app_subtitle_position') as any) || 'bottom');
  const [isGeneratingFinalVideo, setIsGeneratingFinalVideo] = useState(false);
  const [isExportingVideo, setIsExportingVideo] = useState(false);
  const [exportProgress, setExportProgress] = useState(0);
  const [exportSceneInfo, setExportSceneInfo] = useState({ current: 0, total: 0 });
  const stopExportRef = useRef(false);
  const [finalVideoUrl, setFinalVideoUrl] = useState<string | null>(null);
  const [isPreviewingFinal, setIsPreviewingFinal] = useState(false);
  const [currentPreviewIndex, setCurrentPreviewIndex] = useState(0);
  const [isPlayingPreview, setIsPlayingPreview] = useState(false);
  const [isBulkProcessing, setIsBulkProcessing] = useState(false);
  const [bulkMode, setBulkMode] = useState<"images" | "videos" | "all" | null>(null);
  const [bulkProgress, setBulkProgress] = useState<{ current: number; total: number } | null>(null);
  const [isQuotaExceeded, setIsQuotaExceeded] = useState(false);
  const [quotaErrorMessage, setQuotaErrorMessage] = useState<string | null>(null);
  const [consistencyLevel, setConsistencyLevel] = useState<"Standard" | "Extreme">(() => (localStorage.getItem('app_consistency_level') as any) || "Standard");
  const [savedCharacters, setSavedCharacters] = useState<{id: string, name: string, description: string, style: VisualStyle, imageUrl?: string}[]>(() => {
    try {
      const saved = localStorage.getItem('app_saved_characters');
      return saved ? JSON.parse(saved) : [];
    } catch (e) {
      console.error("Error parsing saved characters:", e);
      return [];
    }
  });
  const [story, setStoryState] = useState<Story | null>(null);
  const storyRef = React.useRef<Story | null>(null);
  const saveContextRef = React.useRef<{ user: FirebaseUser | null; currentStoryId: string | null }>({ user: null, currentStoryId: null });

  // Sync helper to ensure ref is always up to date synchronously
  const setStory = (update: Story | null | ((prev: Story | null) => Story | null)) => {
    if (typeof update === 'function') {
      setStoryState(prev => {
        const next = update(prev);
        storyRef.current = next;
        return next;
      });
    } else {
      setStoryState(update);
      storyRef.current = update;
    }
  };

  const buildStoryToSave = (s: Story) => ({
    ...s,
    segments: s.segments.map(seg => {
      const { videoBlob, audioBlob, ...rest } = seg;
      return {
        ...rest,
        videoUrl: seg.videoUrl?.startsWith('blob:') ? null : (seg.videoUrl ?? null),
        audioUrl: seg.audioUrl?.startsWith('blob:') ? null : (seg.audioUrl ?? null),
      };
    }),
    audioUrl: s.audioUrl?.startsWith('blob:') ? null : (s.audioUrl ?? null),
    backgroundMusicUrl: s.backgroundMusicUrl?.startsWith('blob:') ? null : (s.backgroundMusicUrl ?? null),
  });

  useEffect(() => {
    if (story && currentStoryId && user) {
      const timeout = setTimeout(() => {
        updateStoryInDb(buildStoryToSave(story));
      }, 1500);
      return () => clearTimeout(timeout);
    }
  }, [story, currentStoryId, user]);

  // Mantiene refs sincronizados para poder hacer un guardado de emergencia
  // (ver más abajo) sin depender de closures desactualizados.
  useEffect(() => {
    saveContextRef.current = { user, currentStoryId };
  }, [user, currentStoryId]);

  // Guardado de emergencia: si la pestaña se va a cerrar, recargar, o pasa a
  // segundo plano (típico en móviles) mientras hay un guardado pendiente por
  // el debounce de arriba, lo disparamos de inmediato para no perder los
  // últimos cambios/la historia recién generada.
  useEffect(() => {
    const flushPendingSave = () => {
      const { user: u, currentStoryId: csid } = saveContextRef.current;
      const currentStory = storyRef.current;
      if (u && csid && currentStory) {
        updateStoryInDb(buildStoryToSave(currentStory));
      }
    };
    const handleVisibility = () => {
      if (document.visibilityState === 'hidden') flushPendingSave();
    };
    window.addEventListener('beforeunload', flushPendingSave);
    window.addEventListener('pagehide', flushPendingSave);
    document.addEventListener('visibilitychange', handleVisibility);
    return () => {
      window.removeEventListener('beforeunload', flushPendingSave);
      window.removeEventListener('pagehide', flushPendingSave);
      document.removeEventListener('visibilitychange', handleVisibility);
    };
  }, []);

  // Load current story if it changes via settings sync
  useEffect(() => {
    if (user && currentStoryId && !story && isSettingsLoaded) {
      const storyRef = doc(db, 'users', user.uid, 'stories', currentStoryId);
      getDoc(storyRef).then(snap => {
        if (snap.exists()) {
          setStory(snap.data() as Story);
          setIsStoryLoaded(true);
        }
      }).catch(err => {
        handleFirestoreError(err, OperationType.GET, `users/${user.uid}/stories/${currentStoryId}`);
      });
    }
  }, [user, currentStoryId, isSettingsLoaded]);

  const [confirmModal, setConfirmModal] = useState<{
    isOpen: boolean;
    title: string;
    message: string;
    onConfirm: () => void;
    onCancel?: () => void;
    confirmText?: string;
    cancelText?: string;
    type?: 'danger' | 'info';
  }>({
    isOpen: false,
    title: '',
    message: '',
    onConfirm: () => {},
  });

  const requestConfirm = (options: { title: string; message: string; confirmText?: string; cancelText?: string; type?: 'danger' | 'info' }): Promise<boolean> => {
    return new Promise((resolve) => {
      setConfirmModal({
        isOpen: true,
        title: options.title,
        message: options.message,
        confirmText: options.confirmText || "Confirmar",
        cancelText: options.cancelText || "Cancelar",
        type: options.type || "info",
        onConfirm: () => {
          setConfirmModal(prev => ({ ...prev, isOpen: false }));
          resolve(true);
        },
        onCancel: () => {
          setConfirmModal(prev => ({ ...prev, isOpen: false }));
          resolve(false);
        }
      });
    });
  };

  const [showPackagePreview, setShowPackagePreview] = useState(false);
  const [isStoryLoaded, setIsStoryLoaded] = useState(false);
  const [activeSegmentIndex, setActiveSegmentIndex] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [totalCost, setTotalCost] = useState(0);
  const [costBreakdown, setCostBreakdown] = useState({
    stories: 0,
    images: 0,
    videos: 0,
    audios: 0
  });

  useEffect(() => {
    // Force reset cost to 0 for Free Mode and clear storage
    localStorage.setItem('app_total_cost', '0');
    localStorage.setItem('app_cost_breakdown', JSON.stringify({
      stories: 0,
      images: 0,
      videos: 0,
      audios: 0
    }));
  }, []);
  
  const [visualAnchor, setVisualAnchor] = useState<string | null>(() => localStorage.getItem('app_visual_anchor'));
  const [visualAnchorImages, setVisualAnchorImages] = useState<string[]>(() => {
    try {
      const saved = localStorage.getItem('app_visual_anchor_images');
      return saved ? JSON.parse(saved) : [];
    } catch (e) {
      return [];
    }
  });

  // --- Settings Sync with Firestore ---
  useEffect(() => {
    if (!user) {
      setIsSettingsLoaded(false);
      return;
    }

    const settingsRef = doc(db, 'users', user.uid, 'settings', 'current');
    const unsubscribe = onSnapshot(settingsRef, (snapshot) => {
      if (snapshot.exists()) {
        const data = snapshot.data();
        
        // Update states only if they are different to avoid unnecessary re-renders
        if (data.prompt !== undefined) setPrompt(prev => prev !== data.prompt ? data.prompt : prev);
        if (data.characterDesc !== undefined) setCharacterDesc(prev => prev !== data.characterDesc ? data.characterDesc : prev);
        if (data.imageSize !== undefined) setImageSize(prev => prev !== data.imageSize ? data.imageSize : prev);
        if (data.videoAspectRatio !== undefined) setVideoAspectRatio(prev => prev !== data.videoAspectRatio ? data.videoAspectRatio : prev);
        if (data.visualStyle !== undefined) setVisualStyle(prev => prev !== data.visualStyle ? data.visualStyle : prev);
        if (data.dynamicAngles !== undefined) setDynamicAngles(prev => prev !== data.dynamicAngles ? data.dynamicAngles : prev);
        if (data.useExactText !== undefined) setUseExactText(prev => prev !== data.useExactText ? data.useExactText : prev);
        if (data.storyDuration !== undefined) setStoryDuration(prev => prev !== data.storyDuration ? data.storyDuration : prev);
        if (data.modelQuality !== undefined) setModelQuality(prev => prev !== data.modelQuality ? data.modelQuality : prev);
        if (data.selectedVoice !== undefined) setSelectedVoice(prev => prev !== data.selectedVoice ? data.selectedVoice : prev);
        if (data.voiceProvider !== undefined) setVoiceProvider(prev => prev !== data.voiceProvider ? data.voiceProvider : prev);
        if (data.elevenLabsApiKey !== undefined) setElevenLabsApiKey(prev => prev !== data.elevenLabsApiKey ? data.elevenLabsApiKey : prev);
        if (data.selectedElevenLabsVoice !== undefined) setSelectedElevenLabsVoice(prev => prev !== data.selectedElevenLabsVoice ? data.selectedElevenLabsVoice : prev);
        if (data.imageProvider !== undefined) setImageProvider(prev => prev !== data.imageProvider ? data.imageProvider : prev);
        if (data.videoProvider !== undefined) setVideoProvider(prev => prev !== data.videoProvider ? data.videoProvider : prev);
        if (data.burnTextIntoImages !== undefined) setBurnTextIntoImages(prev => prev !== data.burnTextIntoImages ? data.burnTextIntoImages : prev);
        if (data.subtitleStyle !== undefined) setSubtitleStyle(prev => prev !== data.subtitleStyle ? data.subtitleStyle : prev);
        if (data.subtitlePosition !== undefined) setSubtitlePosition(prev => prev !== data.subtitlePosition ? data.subtitlePosition : prev);
        if (data.consistencyLevel !== undefined) setConsistencyLevel(prev => prev !== data.consistencyLevel ? data.consistencyLevel : prev);
        if (data.savedCharacters !== undefined) setSavedCharacters(prev => JSON.stringify(prev) !== JSON.stringify(data.savedCharacters) ? data.savedCharacters : prev);
        if (data.totalCost !== undefined) setTotalCost(prev => prev !== data.totalCost ? data.totalCost : prev);
        if (data.costBreakdown !== undefined) setCostBreakdown(prev => JSON.stringify(prev) !== JSON.stringify(data.costBreakdown) ? data.costBreakdown : prev);
        if (data.visualAnchor !== undefined) setVisualAnchor(prev => prev !== data.visualAnchor ? data.visualAnchor : prev);
        if (data.visualAnchorImages !== undefined) setVisualAnchorImages(prev => JSON.stringify(prev) !== JSON.stringify(data.visualAnchorImages) ? data.visualAnchorImages : prev);
        if (data.manualApiKey !== undefined) setManualApiKey(prev => prev !== data.manualApiKey ? data.manualApiKey : prev);
        if (data.currentStoryId !== undefined) setCurrentStoryId(prev => prev !== data.currentStoryId ? data.currentStoryId : prev);
        if (data.isAuthorized !== undefined) setIsAuthorized(prev => prev !== data.isAuthorized ? data.isAuthorized : prev);
        if (data.apiKeySelected !== undefined) setApiKeySelected(prev => prev !== data.apiKeySelected ? data.apiKeySelected : prev);
        if (data.sidebarTab !== undefined) setSidebarTab(prev => prev !== data.sidebarTab ? data.sidebarTab : prev);
      }
      setIsSettingsLoaded(true);
    }, (err) => {
      handleFirestoreError(err, OperationType.GET, `users/${user.uid}/settings/current`);
      setIsSettingsLoaded(true);
    });

    return () => unsubscribe();
  }, [user]);

  // Guardado INMEDIATO (sin debounce) de currentStoryId: este es el dato crítico
  // que le dice a la app qué historia cargar al volver a entrar. Antes vivía
  // metido en el guardado general de ajustes con 5s de espera, así que si
  // recargabas la página justo después de generar/cambiar de historia (por
  // ejemplo, porque Render acaba de redesplegar un cambio), ese ID nunca
  // llegaba a guardarse y la app "olvidaba" el proyecto en curso.
  useEffect(() => {
    if (!user || !isSettingsLoaded) return;
    const settingsRef = doc(db, 'users', user.uid, 'settings', 'current');
    setDoc(settingsRef, { currentStoryId, updatedAt: Date.now() }, { merge: true }).catch(err => {
      handleFirestoreError(err, OperationType.WRITE, `users/${user.uid}/settings/current`);
    });
  }, [user, isSettingsLoaded, currentStoryId]);

  // Debounced save to Firestore
  useEffect(() => {
    if (!user || !isSettingsLoaded) return;

    const timeout = setTimeout(() => {
      const settingsRef = doc(db, 'users', user.uid, 'settings', 'current');
      const cleanSettings = sanitizeForFirestore({
        prompt,
        characterDesc,
        imageSize,
        videoAspectRatio,
        visualStyle,
        dynamicAngles,
        useExactText,
        storyDuration,
        modelQuality,
        selectedVoice,
        voiceProvider,
        elevenLabsApiKey,
        selectedElevenLabsVoice,
        imageProvider,
        videoProvider,
        burnTextIntoImages,
        subtitleStyle,
        subtitlePosition,
        consistencyLevel,
        savedCharacters,
        totalCost,
        costBreakdown,
        visualAnchor,
        visualAnchorImages,
        manualApiKey,
        isAuthorized,
        apiKeySelected,
        sidebarTab,
        updatedAt: Date.now()
      });
      setDoc(settingsRef, cleanSettings, { merge: true }).catch(err => {
        handleFirestoreError(err, OperationType.WRITE, `users/${user.uid}/settings/current`);
      });
    }, 5000); // 5s debounce for settings (no crítico: prompt, preferencias, etc. — currentStoryId ya se guarda aparte e inmediato arriba)

    return () => clearTimeout(timeout);
  }, [
    user, isSettingsLoaded, prompt, characterDesc, imageSize, videoAspectRatio, visualStyle,
    dynamicAngles, useExactText, storyDuration, modelQuality, selectedVoice,
    imageProvider, videoProvider, burnTextIntoImages, subtitleStyle,
    subtitlePosition, consistencyLevel, savedCharacters, totalCost,
    costBreakdown, visualAnchor, visualAnchorImages, manualApiKey,
    isAuthorized, apiKeySelected, sidebarTab
  ]);

  async function withRetry<T>(fn: () => Promise<T>, maxRetries = 5, delay = 3000): Promise<T> {
    let lastError: any;
    for (let i = 0; i < maxRetries; i++) {
      try {
        return await fn();
      } catch (err: any) {
        lastError = err;
        const errorStr = JSON.stringify(err).toLowerCase();
        const msg = String(err.message || "").toLowerCase();
        const status = String(err.status || "").toLowerCase();
        
        const isRetryable = msg.includes("503") || 
                            msg.includes("502") ||
                            msg.includes("504") ||
                            msg.includes("500") ||
                            msg.includes("internal error") ||
                            msg.includes("429") || 
                            msg.includes("high demand") ||
                            msg.includes("unavailable") ||
                            msg.includes("deadline exceeded") ||
                            msg.includes("bad gateway") ||
                            msg.includes("gateway timeout") ||
                            status.includes("unavailable") ||
                            status.includes("internal") ||
                            errorStr.includes("503") ||
                            errorStr.includes("502") ||
                            errorStr.includes("504") ||
                            errorStr.includes("500") ||
                            errorStr.includes("internal") ||
                            errorStr.includes("unavailable");
        
        const isPaidKey = getApiKeySource() === 'Manual' || getApiKeySource() === 'W-P-E-A';
        const effectiveMaxRetries = isPaidKey ? maxRetries + 3 : maxRetries;

        if (isRetryable && i < effectiveMaxRetries - 1) {
          const isInternal = msg.includes("500") || msg.includes("internal") || status.includes("internal");
          const retryLimit = isInternal ? effectiveMaxRetries + 2 : effectiveMaxRetries;
          
          if (i < retryLimit - 1) {
            console.warn(`Retrying after error: ${err.message || "Unknown error"}. Attempt ${i + 1}/${retryLimit}`);
            // Exponential backoff: 3s, 6s, 12s, 24s...
            await new Promise(resolve => setTimeout(resolve, delay * Math.pow(2, i)));
            continue;
          }
        }

        if (msg.includes("429") || errorStr.includes("429") || msg.includes("quota") || errorStr.includes("quota")) {
          setIsQuotaExceeded(true);
          const quotaMsg = isPaidKey 
            ? "Has alcanzado el límite de tu cuota de pago o de velocidad. Por favor, revisa tu consola de Google Cloud Billing."
            : "Has agotado el límite gratuito de Google Gemini (Error 429 / Quota Exceeded). El límite suele resetearse cada minuto para peticiones de texto y cada día para cuotas totales. Por favor, espera un momento o usa una API Key propia en Ajustes.";
          setQuotaErrorMessage(quotaMsg);
          throw new Error(quotaMsg);
        }

        if (msg.includes("503") || errorStr.includes("503")) {
          const paidMsg = isPaidKey 
            ? "Los servidores de Google están bajo una carga extrema. Aunque tienes una clave de pago activa (Modo Premium), incluso su infraestructura tiene límites físicos ocasionales. Estamos reintentando con máxima prioridad, pero por favor espera unos segundos e inténtalo de nuevo."
            : "Los servidores de Google están saturados (Error 503). Hemos intentado reintentar automáticamente, pero la demanda sigue siendo muy alta. Por favor, espera un minuto e inténtalo de nuevo.";
          throw new Error(paidMsg);
        }
        throw err;
      }
    }
    throw lastError;
  }

  // --- Constants & Types ---

const DB_NAME = "StoryAppDB";
const STORE_NAME = "stories";
const STORY_KEY = "current_story";

async function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME);
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function saveStoryToDB(story: Story) {
  try {
    const db = await openDB();
    const tx = db.transaction(STORE_NAME, "readwrite");
    const store = tx.objectStore(STORE_NAME);
    store.put(story, STORY_KEY);
    return new Promise((resolve, reject) => {
      tx.oncomplete = () => resolve(true);
      tx.onerror = () => reject(tx.error);
    });
  } catch (err) {
    console.error("Failed to save story to IndexedDB:", err);
  }
}

async function loadStoryFromDB(): Promise<Story | null> {
  try {
    const db = await openDB();
    const tx = db.transaction(STORE_NAME, "readonly");
    const store = tx.objectStore(STORE_NAME);
    const request = store.get(STORY_KEY);
    return new Promise((resolve, reject) => {
      request.onsuccess = () => {
        const story = request.result as Story | null;
        if (story) {
          // Recreate blob URLs for segments
          story.segments = story.segments.map(seg => {
            if (seg.videoBlob) {
              seg.videoUrl = URL.createObjectURL(seg.videoBlob);
            }
            if (seg.audioBlob) {
              // If we have per-segment audio
            }
            return seg;
          });

          // Recreate main audio URL
          if (story.audioBlob) {
            story.audioUrl = URL.createObjectURL(story.audioBlob);
          }

          // Recreate background music URL
          if (story.backgroundMusicBlob) {
            story.backgroundMusicUrl = URL.createObjectURL(story.backgroundMusicBlob);
          }
        }
        resolve(story);
      };
      request.onerror = () => reject(request.error);
    });
  } catch (err) {
    console.error("Failed to load story from IndexedDB:", err);
    return null;
  }
}

const COSTS = {
    STORY_GEN: 0,
    IMAGE: {
      "1K": 0,
      "2K": 0,
      "4K": 0
    },
    VIDEO_GEN: 0,
    AUDIO_GEN: 0
  };

  const addCost = (amount: number, type: keyof typeof costBreakdown) => {
    setTotalCost(prev => prev + amount);
    setCostBreakdown(prev => ({ ...prev, [type]: prev[type] + 1 }));
  };
  
  const [detectedKey, setDetectedKey] = useState<string>('');

  useEffect(() => {
    const loadInitialStory = async () => {
      try {
        // 1. Try Firestore if user is logged in and we have a currentStoryId
        if (user && currentStoryId) {
          const storyRef = doc(db, 'users', user.uid, 'stories', currentStoryId);
          const storySnap = await getDoc(storyRef);
          if (storySnap.exists()) {
            const fsStory = storySnap.data() as Story;
            // Reset generation flags
            fsStory.segments = (fsStory.segments || []).map(s => ({
              ...s,
              isGeneratingImage: false,
              isGeneratingVideo: false
            }));
            setStory(fsStory);
            setIsStoryLoaded(true);
            return;
          }
        }

        // 2. Fallback to IndexedDB
        const dbStory = await loadStoryFromDB();
        if (dbStory) {
          dbStory.segments = (dbStory.segments || []).map(s => ({
            ...s,
            isGeneratingImage: false,
            isGeneratingVideo: false
          }));
          setStory(dbStory);
        } else {
          // 3. Fallback to localStorage
          const saved = localStorage.getItem('app_current_story');
          if (saved) {
            try {
              const parsed = JSON.parse(saved);
              if (parsed && parsed.segments) {
                parsed.segments = parsed.segments.map((s: any) => ({
                  ...s,
                  isGeneratingImage: false,
                  isGeneratingVideo: false
                }));
              }
              setStory(parsed);
            } catch (e) {
              console.error("Failed to parse localStorage story", e);
            }
          }
        }
      } catch (err) {
        console.error("Error loading initial story:", err);
      } finally {
        setIsStoryLoaded(true);
      }
    };
    
    if (isSettingsLoaded) {
      loadInitialStory();
    }
  }, [user, currentStoryId, isSettingsLoaded]);

  useEffect(() => {
    if (!isStoryLoaded) return;
    const saveState = async () => {
      try {
        const stateToSave = {
          isAuthorized,
          apiKeySelected,
          prompt,
          characterDesc,
          imageSize,
          videoAspectRatio,
          visualStyle,
          subtitleStyle,
          subtitlePosition,
          dynamicAngles,
          useExactText,
          storyDuration,
          modelQuality,
          selectedVoice,
          imageProvider,
          videoProvider,
          totalCost,
          costBreakdown,
          savedCharacters,
          visualAnchor,
          visualAnchorImages
        };
        
        Object.entries(stateToSave).forEach(([key, value]) => {
          const storageKey = key.startsWith('is_') || key.startsWith('api_') ? key : `app_${key.replace(/[A-Z]/g, letter => `_${letter.toLowerCase()}`)}`;
          if (value !== null && value !== undefined) {
            localStorage.setItem(storageKey, typeof value === 'object' ? JSON.stringify(value) : String(value));
          } else {
            localStorage.removeItem(storageKey);
          }
        });

        if (story) {
          // Save to IndexedDB for large data (images)
          await saveStoryToDB(story);
          
          // Save a lightweight version to localStorage as backup/sync
          const storyWithoutImages = {
            ...story,
            segments: story.segments.map(s => ({ 
              ...s, 
              imageUrl: undefined, 
              videoUrl: undefined,
              videoBlob: undefined,
              audioBlob: undefined
            })),
            audioBlob: undefined,
            backgroundMusicBlob: undefined
          };
          localStorage.setItem('app_current_story', JSON.stringify(storyWithoutImages));
        } else {
          localStorage.removeItem('app_current_story');
          // Clear IndexedDB story
          const db = await openDB();
          const tx = db.transaction(STORE_NAME, "readwrite");
          tx.objectStore(STORE_NAME).delete(STORY_KEY);
        }
      } catch (err) {
        console.error("Error saving state:", err);
      }
    };

    saveState();
  }, [
    isAuthorized, apiKeySelected, prompt, characterDesc, imageSize, 
    videoAspectRatio, visualStyle, dynamicAngles, useExactText, storyDuration, 
    modelQuality, selectedVoice, imageProvider, videoProvider, story, totalCost, costBreakdown, 
    savedCharacters, visualAnchor, visualAnchorImages, isStoryLoaded
  ]);

  useEffect(() => {
    const interval = setInterval(() => {
      const key = getApiKey();
      const source = getApiKeySource();
      const hasKey = key || source !== 'Ninguna';
      const effectiveKey = key || (hasKey ? 'DETECTED' : '');
      if (effectiveKey !== detectedKey) {
        setDetectedKey(effectiveKey);
      }
    }, 1000);
    return () => clearInterval(interval);
  }, [detectedKey, modelQuality, manualApiKey, imageProvider, elevenLabsApiKey, videoProvider]);

  useEffect(() => {
    // Check for aistudio availability with retries
    const checkReady = () => {
      if (window.aistudio) {
        setIsAiStudioReady(true);
        checkApiKey();
        return true;
      }
      return false;
    };

    if (!checkReady()) {
      const interval = setInterval(() => {
        if (checkReady()) clearInterval(interval);
      }, 500);
      return () => clearInterval(interval);
    }

    // Load saved characters - now handled by Firestore sync if logged in
    if (!user) {
      const stored = localStorage.getItem('story_visualizer_characters');
      if (stored) {
        try {
          setSavedCharacters(JSON.parse(stored));
        } catch (e) {
          console.error("Failed to load characters", e);
        }
      }
    }
  }, [user]);

  const checkApiKey = async () => {
    try {
      if (window.aistudio?.hasSelectedApiKey) {
        const hasKey = await window.aistudio.hasSelectedApiKey();
        // Only consider the key "selected" if the platform says so AND we can actually read it
        // This prevents entering the app in a state where AI calls will fail
        if (hasKey && getApiKey()) {
          setApiKeySelected(true);
        }
      }
    } catch (err) {
      console.error("Error checking API key:", err);
    }
  };

  const handleSelectKey = async () => {
    try {
      if (window.aistudio?.openSelectKey) {
        await window.aistudio.openSelectKey();
        // As per guidelines, assume success and proceed
        setApiKeySelected(true);
      } else {
        // Fallback if the platform tool is missing
        console.warn("AI Studio openSelectKey not found");
        setApiKeySelected(true); 
      }
    } catch (err) {
      console.error("Error opening key selector:", err);
      setApiKeySelected(true); // Proceed anyway to not block the user
    }
  };

  const getApiKey = () => {
    // 0. Check for manually entered key
    if (manualApiKey && manualApiKey.trim().startsWith('AIzaSy')) {
      return manualApiKey.trim();
    }

    // 1. Check for paid key injected by platform (API_KEY)
    let key = getVal(window, 'process.env.API_KEY') || 
              getVal(window, 'API_KEY') ||
              (window as any)['API_KEY'] ||
              getVal(globalThis, 'process.env.API_KEY') ||
              getVal(globalThis, 'API_KEY');
    
    // 2. Check for GEMINI_API_KEY (Secrets tab or env)
    if (!key || typeof key !== 'string' || key.length < 5) {
      key = getVal(window, 'process.env.GEMINI_API_KEY') || 
            getVal(window, 'GEMINI_API_KEY') ||
            (window as any)['GEMINI_API_KEY'] ||
            getVal(globalThis, 'process.env.GEMINI_API_KEY') ||
            getVal(globalThis, 'GEMINI_API_KEY') ||
            (import.meta as any).env?.VITE_GEMINI_API_KEY ||
            (import.meta as any).env?.GEMINI_API_KEY;
    }

    // 3. Check for any other common names
    if (!key || typeof key !== 'string' || key.length < 5) {
      key = getVal(window, 'process.env.VITE_GEMINI_API_KEY') ||
            getVal(window, 'VITE_GEMINI_API_KEY');
    }
    
    // 3. Final validation
    if (!key || typeof key !== 'string') return '';
    
    const trimmedKey = key.trim();
    const invalidValues = ['undefined', 'null', '', 'MY_GEMINI_API_KEY', 'TODO_KEYHERE'];
    if (invalidValues.includes(trimmedKey)) {
      return '';
    }
    
    return trimmedKey;
  };

  const getApiKeySource = () => {
    if (manualApiKey && manualApiKey.trim().startsWith('AIzaSy')) return 'Manual';
    if (getVal(window, 'process.env.API_KEY')) return 'W-P-E-A';
    if (getVal(window, 'API_KEY')) return 'W-A';
    if (getVal(window, 'process.env.GEMINI_API_KEY')) return 'W-P-E-G';
    if (getVal(window, 'GEMINI_API_KEY')) return 'W-G';
    if ((import.meta as any).env?.VITE_GEMINI_API_KEY) {
      const k = (import.meta as any).env?.VITE_GEMINI_API_KEY;
      if (k && k !== 'MY_GEMINI_API_KEY') return 'V-G-A';
    }
    if (elevenLabsApiKey && elevenLabsApiKey.trim().length > 10) return 'ElevenLabs';
    if (imageProvider === "Pollinations") return 'Pollinations (Gratis)';
    if (videoProvider === "CinematicPan") return 'Cinematic Pan (Gratis)';
    return 'Ninguna';
  };

  // Helper to get value by path safely
  function getVal(obj: any, path: string) {
    try {
      return path.split('.').reduce((acc, part) => acc && acc[part], obj);
    } catch {
      return undefined;
    }
  }

  const handleManualApiKeySave = (e: React.FormEvent) => {
    e.preventDefault();
    localStorage.setItem('manual_gemini_api_key', manualApiKey);
    setShowApiKeyInput(false);
    setApiKeySelected(true);
  };

  const updateStoryInDb = async (updatedStory: Story) => {
    if (!user || !currentStoryId) return;
    try {
      const storyRef = doc(db, 'users', user.uid, 'stories', currentStoryId);
      const cleanData = sanitizeForFirestore({
        ...updatedStory,
        userId: user.uid,
        updatedAt: Date.now()
      });
      await setDoc(storyRef, cleanData, { merge: true });
    } catch (err) {
      handleFirestoreError(err, OperationType.UPDATE, `users/${user.uid}/stories/${currentStoryId}`);
    }
  };

  const resetProject = async () => {
    const confirmed = await requestConfirm({
      title: "Nuevo Proyecto",
      message: "¿Estás seguro de que quieres empezar un nuevo proyecto? Se borrará el progreso actual y todos los assets generados.",
      confirmText: "Empezar de Nuevo",
      type: "danger"
    });

    if (!confirmed) return;

    setStory(null);
    setCurrentStoryId(null);
    setPrompt('');
    setCharacterDesc('');
    setVisualAnchor(null);
    setVisualAnchorImages([]);
    setActiveSegmentIndex(0);
    setError(null);
    localStorage.removeItem('app_current_story');
    localStorage.removeItem('app_prompt');
    localStorage.removeItem('app_character_desc');
    localStorage.removeItem('app_visual_anchor');
    localStorage.removeItem('app_visual_anchor_images');
    
    // Clear Firestore if user is logged in
    if (user) {
      const settingsRef = doc(db, 'users', user.uid, 'settings', 'current');
      const cleanSettings = sanitizeForFirestore({
        currentStoryId: null,
        prompt: '',
        characterDesc: '',
        visualAnchor: null,
        visualAnchorImages: [],
        updatedAt: Date.now()
      });
      setDoc(settingsRef, cleanSettings, { merge: true }).catch(err => handleFirestoreError(err, OperationType.UPDATE, `users/${user.uid}/settings/current`));
    }

    // Clear IndexedDB
    openDB().then(db => {
      const tx = db.transaction(STORE_NAME, "readwrite");
      tx.objectStore(STORE_NAME).delete(STORY_KEY);
    });
  };

  const handlePasswordSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (passwordInput === '041708') {
      setIsAuthorized(true);
      setPasswordError(false);
    } else {
      setPasswordError(true);
    }
  };

  const validateApiKey = (feature: 'story' | 'visual' | 'audio' | 'video' = 'story'): string | null => {
    // Story generation now runs on Groq, configured server-side. No client key needed.
    if (feature === 'story') {
      return null;
    }

    // For visual features, if using Pollinations, we don't need a Gemini key
    if (feature === 'visual' && imageProvider === "Pollinations") {
      return null;
    }

    // For audio features, if using ElevenLabs, we don't need a Gemini key
    if (feature === 'audio' && elevenLabsApiKey && elevenLabsApiKey.trim().length > 10) {
      return null;
    }

    // For video features, if using CinematicPan, we don't need a Gemini key
    if (feature === 'video' && videoProvider === "CinematicPan") {
      return null;
    }

    const key = getApiKey();
    if (!key) {
      if (feature === 'visual') {
        return "Para generar imágenes gratis, selecciona 'Proveedor: Pollinations' en los ajustes. Si prefieres usar Google AI, configura tu API Key.";
      } else if (feature === 'audio') {
        return "La generación de audio (Narración) requiere una API Key de Gemini o ElevenLabs. Configúrala en los ajustes.";
      } else if (feature === 'video') {
        return "La generación de video requiere una API Key de Gemini. Configúrala en los ajustes.";
      }
      return "Se requiere una API Key para esta función. Configúrala en los ajustes.";
    }
    
    if (!key.startsWith('AIzaSy')) {
      return "La clave API detectada no parece ser válida para Google AI (debe comenzar con 'AIzaSy'). Por favor, verifica tu configuración.";
    }
    return null;
  };

  const handleAvatarUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      const reader = new FileReader();
      reader.onloadend = () => {
        const base64 = reader.result as string;
        setVisualAnchorImages(prev => [...prev, base64]);
      };
      reader.readAsDataURL(file);
    }
  };

  const deleteAvatar = (index: number) => {
    setVisualAnchorImages(prev => prev.filter((_, i) => i !== index));
  };

  const generateStory = async (overridePrompt?: string, overrideStyle?: VisualStyle, type: 'quote' | 'narrative' | 'analyze' = 'narrative') => {
    const activePrompt = overridePrompt || prompt;
    const activeStyle = overrideStyle || visualStyle;

    if (!activePrompt.trim()) {
      setError("Por favor, escribe un concepto para la historia.");
      return null;
    }
    
    const validationError = validateApiKey();
    if (validationError) {
      setError(validationError);
      return null;
    }

    setIsGeneratingStory(true);
    setError(null);
    
    try {
      const segmentCounts = {
        "Auto": "tantos como sean necesarios para cubrir toda la historia en detalle (sin límite fijo)",
        "30s": "5-6",
        "1m": "10-12",
        "2m": "18-22",
        "3m": "25-30",
        "10imgs": "exactamente 10",
        "20imgs": "exactamente 20",
        "30imgs": "exactamente 30",
        "40imgs": "exactamente 40",
        "50imgs": "exactamente 50",
        "100imgs": "exactamente 100"
      };

      const isReflectionMode = type === 'quote' || storyDuration.includes("imgs") || activeStyle.includes("Reflection");
      
      let styleDetail = "";
      if (activeStyle === "3D Animation") styleDetail = "Masterpiece, High-end 3D Animation style, Pixar or Disney aesthetic, extremely high quality, expressive characters with large eyes and animated proportions, detailed but stylized textures, vibrant lighting, soft rendering, 8k, cinematic camera, colorful and lively atmosphere, smooth gradients.";
      else if (activeStyle === "Reflection Dark") styleDetail = "Masterpiece, dark, nocturnal, dramatic, deep shadows, volumetric lighting, 8k, cinematic, high contrast, vibrant highlights";
      else if (activeStyle === "Reflection Nature") styleDetail = "Masterpiece, epic natural landscapes, dense forests, majestic mountains, crystal clear lakes, golden sunrises, ultra detailed, 8k, vibrant natural colors, cinematic lighting";
      else if (activeStyle === "Reflection Urban") styleDetail = "Masterpiece, vibrant cities, rainy streets with reflections, neon lights, detailed modern architecture, cinematic style, 8k, rich color palette, wet surfaces";
      else if (activeStyle === "Gold Skeleton") styleDetail = "Masterpiece, solid gold human skeleton, shiny, hyper-realistic, 8k, cinematic lighting, liquid gold textures, glowing golden aura, high reflections";
      else if (activeStyle === "Glass Skeleton") styleDetail = "Masterpiece, translucent glass human skeleton, visible internal organs, glowing core, hyper-realistic, 8k, cinematic lighting, glass and gold textures, refractive surfaces, vibrant energy";
      else if (activeStyle === "Cyber-Schematic") styleDetail = "Masterpiece, Cyber-Schematic Wireframe, Holographic Combat HUD, technical polygonal mesh, Amber and Cyan high-contrast, HUD interface elements, electric sparks, digital glitch effects";
      else if (activeStyle === "Manhwa Premium") styleDetail = "Masterpiece, Supreme Quality, Ultra detailed premium mature manhwa style, semi-realistic digital illustration, perfect airbrushed glowing skin, symmetrical hyper-idealized features, cinematic rim lighting, vibrant saturated colors. FEMALE CHARACTER: supreme beauty, extremely attractive, hyper-idealized exaggerated hourglass figure, large lifted bust, narrow waist, wide hips, thick thighs, sculpted athletic body, long flowing silky hair, alluring and seductive expression. MALE CHARACTER: supreme handsomeness, massive build, wide shoulders, broad chest, large pectorals, alpha male physique, sharp jawline, intense gaze, rugged handsomeness.";
      else if (activeStyle === "Neon Blueprint") styleDetail = "Masterpiece, hyper-detailed scene where the ENTIRE WORLD is built from a SINGLE UNIFIED dense glowing wireframe mesh. Humanoid figures, buildings, streets, furniture, and trees must all be part of the same glowing wireframe structure. NO HUMAN SKIN, NO SOLID FABRICS, only glowing neon wireframe. Characters' bodies, hair, and ALL THEIR CLOTHING MUST BE RENDERED IN THEIR EXACT REFERENCE COLOR. No white clothing, no black clothing, only clothing made of glowing wireframe matching their skin color. The characters wear Casual Streetwear (jackets, jeans, sneakers) that is fully integrated into the glowing wireframe mesh, rendered in their exact color. Environment: A 3D technical blueprint of a real-world everyday urban setting (street, cafe, park) where every object is rendered as a detailed dark glowing wireframe (e.g., dark blue) to contrast with the characters. STRICTLY PROHIBITED: NO NUMBERS, NO TEXT, NO LETTERS, NO SIGNS, NO CARS, NO SOLID CLOTHING. Monochromatic character lighting, uniform clean 3D viewport grid topology, high contrast, 8k.";
      else if (activeStyle === "Neon Pro") styleDetail = "Masterpiece, hyper-detailed scene. Characters are made entirely of a highly detailed, glowing neon polygonal wireframe mesh. Very realistic human anatomy and proportions despite being made of glowing wireframe. ALL CLOTHING MUST MATCH THE CHARACTER'S EXACT AVATAR COLOR (e.g., if the character is pink wireframe, their jacket and pants MUST also be pink wireframe). Dark solid background to emphasize the glowing wireframe characters without the background being wireframe. STRICTLY PROHIBITED: NO NUMBERS, NO TEXT, NO SIGNS, NO CARS, NO VEHICLES. Monochromatic character lighting, uniform clean 3D viewport grid topology, volumetric glow, high contrast, 8k.";
      else if (activeStyle === "HOLOGRAPHIC SOUL") styleDetail = "Masterpiece, hyper-detailed scene. The character is a solid, glowing, translucent volumetric light silhouette, like a being made of soft ethereal energy. Smooth diffuse glow, no wireframe grid lines, no visible skin or fabric texture. Dark, minimalist, almost black background to make the glowing figure stand out. Subtle chromatic soft-focus edges, gentle particle glow, 8k, ethereal and contemplative atmosphere.";
      else if (activeStyle === "Power Couple") styleDetail = "Masterpiece, Supreme Quality, High-end digital illustration, luxury mature comic style, premium graphic novel aesthetic. Elegant strokes, dramatic volumetric shading, high-contrast lighting. CHARACTERS: Stunningly beautiful, supreme attractiveness, hyper-idealized exaggerated hourglass physiques, seductive poses, vibrant colors, sophisticated atmosphere.";
      else if (activeStyle === "Comic Realista Moderno") styleDetail = "Masterpiece, Supreme Quality, High-end Realistic Graphic Novel style, clean sharp black outlines, dramatic chiaroscuro lighting, high-contrast. CHARACTERS: Hyper-idealized and sculptural anatomy, extremely attractive and seductive. FEMALE: Hyper-idealized exaggerated hourglass figure, large lifted bust, narrow waist, wide hips, thick thighs, sculpted athletic body, long flowing silky hair, alluring and seductive expression. MALE: Massive build, wide shoulders, broad chest, large pectorals, alpha male physique, sharp jawline, intense gaze. SCENES: Cinematic composition, deep shadows, sophisticated color palette, high-definition textures, 8k, octane render style.";
      else if (activeStyle === "Dios") styleDetail = "Masterpiece, High-end Divine and Majestic style. Cinematic lighting with volumetric light rays (god rays) piercing through clouds or windows. Celestial and biblical figures with subtle but powerful auras. Dramatic contrast between sacred lights and deep shadows. Hyper-realistic textures on fabrics and skin. Monumental settings: infinite skies, ancient temples, or modern environments with a supernatural atmosphere. 8k quality, epic and emotional. Characters must have an imposing and serene presence. IF THE CHARACTER IS JESUS OR CHRIST, he must have his traditional iconic appearance: long brown hair, well-groomed beard, humble but majestic linen robes, and a look of compassion and deep wisdom.";
      else if (activeStyle === "Realismo Crudo") styleDetail = "Masterpiece, Raw Realism and Documentary style. Ordinary people, non-symmetric faces, natural skin texture, visible pores, slight imperfections, tired eyes, realistic proportions. Documentary photography aesthetic, NOT heroic, NOT model-like, no celebrity likeness, no perfect face. Naturalistic lighting, often raw or ambient light, capturing the essence of daily life without idealizing filters.";
      else if (activeStyle === "Minimal Clay Animation") styleDetail = "Masterpiece, minimalist, 3D clay animation style, matte clay material finish. Characters are solid, matte white clay figures with smooth, unified surfaces. ABSOLUTELY NO neon, NO wireframes, NO glowing effects, NO transparency, NO holographic materials. The figure must be uniform solid matte clay color (skin and clothes). Smooth, soft clay texture. Minimalist clean backgrounds, soft-box studio lighting, soft shadows, 8k render, minimalist toy figurine aesthetic.";
      else if (activeStyle === "Oil Painting") styleDetail = "Masterpiece, High-end Oil Painting style, thick brushstrokes, rich textures, vibrant colors, classical lighting, artistic and expressive.";
      else if (activeStyle === "Comic Book") styleDetail = "Masterpiece, High-end Comic Book style, bold lines, vibrant colors, dynamic shading, cinematic composition, attractive characters.";
      else if (activeStyle === "Anime") styleDetail = "Masterpiece, High-end Mature Anime style, vibrant colors, clean lines, expressive eyes, cinematic lighting, detailed backgrounds, aesthetic and attractive characters with exaggerated hourglass figures.";
      else if (activeStyle === "Cinematic") styleDetail = "Masterpiece, Supreme Quality, High-end Cinematic photorealistic style, 8k, dramatic lighting, vibrant colors, shallow depth of field, professional color grading, anamorphic lens flares, highly detailed skin textures, extremely attractive characters.";
      else styleDetail = "Masterpiece, minimalist, melancholic or inspiring, with great attention to atmosphere and textures, vibrant colors where appropriate";
      
      const reflectionInstruction = isReflectionMode 
        ? `Esta es una serie de imágenes de reflexión independientes. Cada segmento debe ser una escena ÚNICA y COMPLETAMENTE DISTINTA de las demás. 
           IMPORTANTE: Varía los personajes (diferentes etnias, edades, géneros si es apropiado), los entornos (interior, exterior, naturaleza, ciudad) y las composiciones (plano detalle, plano general, contrapicado).
           The visual style must be: ${styleDetail}. Describe the background with EXTREME DETAIL (weather, lighting, objects, atmosphere). DO NOT include text inside the image. STRICTLY PROHIBITED: Do not include character names or any text written in the scene.`
        : `The visual description ('imagePrompt') must be EXTREMELY DETAILED (minimum 150 words per segment) and written in ENGLISH. The visual style MUST be: ${styleDetail}. It must focus on the action, environment, lighting, and ALWAYS reference the main character to maintain narrative consistency. Define a UNIQUE, DETAILED, and SUPREME BEAUTY character design for this story (ethnicity, perfect features, athletic build, luxurious style) to avoid generic or repetitive characters. 
           IMPORTANT: Vary the clothing and environment in each segment so the story progresses visually. Describe the character's clothing and their exact HAIR COLOR AND STYLE in each segment to maintain absolute consistency. Describe the background, depth of field, and atmosphere of the scene. STRICTLY PROHIBITED: Do not include character names or any text written in the scene.`;

      const dynamicAnglesInstruction = dynamicAngles 
        ? `IMPORTANTE: Usa ángulos de cámara variados y creativos para cada segmento para que el video sea dinámico. No repitas el mismo ángulo más de dos veces en toda la historia. Elige entre estos ángulos para cada 'cameraAngle': 
           - 'Close-up': Primer plano del rostro para captar emociones intensas.
           - 'Extreme Close-up': Primer plano extremo de un detalle (ojos, manos, un objeto clave).
           - 'Conversation': Toma de plano medio, ideal para diálogos o interacciones.
           - 'Aerial': Toma aérea de ángulo alto para mostrar la escala del entorno.
           - 'Wide Shot': Toma panorámica que sitúa al personaje en su contexto.
           - 'Fisheye': Lente gran angular con distorsión para escenas de acción o tensión.
           - 'Low Angle': Toma desde abajo mirando hacia arriba para dar heroísmo o poder.
           - 'Dutch Angle': Plano inclinado para generar inquietud, caos o dinamismo.
           - 'Mirror Reflection': El personaje visto a través de un reflejo.
           - 'Standard': Toma cinematográfica equilibrada.
           Asigna uno de estos a cada segmento en el campo 'cameraAngle' y asegúrate de que el 'imagePrompt' refleje ese ángulo.`
        : "Usa ángulos de cámara cinematográficos estándar para todos los segmentos.";

      const numericSegmentCounts: Record<StoryDuration, number> = {
        "Auto": 12,
        "30s": 5,      // 5 images for 30s (~6s per img)
        "1m": 12,      // 12 images for 1m (~5s per img)
        "2m": 24,      // 24 images for 2m
        "3m": 36,      // 36 images for 3m
        "10imgs": 10,
        "20imgs": 20,
        "30imgs": 30,
        "40imgs": 40,
        "50imgs": 50,
        "100imgs": 100
      };

      const wordLimits: Record<StoryDuration, number> = {
        "Auto": 300,
        "30s": 75,
        "1m": 150,     // 150 words for 1 min (average reading speed)
        "2m": 300,
        "3m": 450,
        "10imgs": 250,
        "20imgs": 500,
        "30imgs": 750,
        "40imgs": 1000,
        "50imgs": 1250,
        "100imgs": 2500
      };

      const segmentsToGenerate = Math.min(numericSegmentCounts[storyDuration] || 10, 50); // Aumentado el límite a 50 para mayor flexibilidad
      const promptType = type === 'quote' ? "FRASES RÁPIDAS / QUOTES" : "HISTORIA NARRATIVA";
      
      const durationInstruction = storyDuration === "Auto" 
        ? `Cubre TODA la historia en detalle, sin omitir partes importantes. Decide tú mismo cuánto debe durar la narración según lo que el tema realmente necesite para completarse bien (normalmente entre 1 y 3 minutos para contenido de redes sociales, salvo que el tema pida claramente más). 
           CÁLCULO DE SEGMENTOS (OBLIGATORIO): Una vez decidida la duración de la narración, DIVÍDELA en aproximadamente 10 a 11 segmentos POR CADA MINUTO de narración (ejemplos: ~1:40 min → entre 15 y 18 segmentos; ~1:00 min → 10-12 segmentos; ~2:00 min → 20-22 segmentos). Cada segmento debe cubrir una escena o "beat" narrativo completo (aprox 12-18 palabras) — ESTÁ PROHIBIDO fragmentar una sola escena o emoción en varios segmentos casi idénticos solo para sumar más cantidad. NUNCA generes más de 22 segmentos en modo automático salvo que el tema sea excepcionalmente extenso y realmente lo amerite.`
        : `ES VITAL PARA EL RITMO DEL AUDIO: Tu narración COMPLETA (texto) debe tener EXACTAMENTE EN TOTAL unas ${wordLimits[storyDuration]} palabras (para durar el tiempo exacto). DÍVIDE este texto en EXACTAMENTE ${segmentsToGenerate} segmentos. Cada segmento DEBE TENER entre 12 y 15 palabras como MÁXIMO. ¡PROHIBIDO hacer segmentos largos! Esto asegurará matemáticamente generar 1 imagen por cada ~5 segundos de audio (mínimo 10 por minuto).`;

      const characterIdentificationInstruction = `IDENTIFICACIÓN DE PERSONAJE POR ESCENA (OBLIGATORIO, APLICA A TODAS LAS CATEGORÍAS): Si la historia tiene más de un personaje (ej. un hombre y una mujer, o varios personajes con nombre propio), cada 'imagePrompt' y cada 'videoDescription' DEBE dejar explícito y sin ambigüedad CUÁL personaje aparece en esa escena — usa términos claros como "the man"/"el hombre", "the woman"/"la mujer", o el nombre propio del personaje si la historia se lo dio, en vez de términos genéricos y ambiguos como "the character", "la figura" o "el personaje" cuando pueda haber más de uno. Si la escena incluye a ambos personajes a la vez, dilo explícitamente (ej. "el hombre y la mujer, uno frente al otro"). Si la historia tiene UN SOLO personaje en total, sí puedes usar "el personaje"/"the character" de forma consistente en todos los segmentos.`;

      const neonBlueprintInstruction = activeStyle === "Neon Blueprint" || activeStyle === "Neon Pro"
        ? `ESPECIAL PARA ${activeStyle.toUpperCase()}: TODO en la imagen (personajes, ropa, calzado${activeStyle === "Neon Blueprint" ? " y TODO EL ENTORNO" : ""}) debe ser una malla de neón brillante (glowing wireframe mesh). 
           IMPORTANTE: Los personajes DEBEN llevar calzado (sneakers, tacones) hecho de malla de neón. NO DEBEN ESTAR DESCALZOS.
           COLORES CROMÁTICOS SEPARADOS Y CONSISTENCIA: Si el personaje tiene un COLOR ESPECÍFICO en la referencia visual (ejemplo: azul brillante, o naranja), entonces SU CUERPO DE MALLA Y SU ROPA DE MALLA DEBEN SER DE ESE MISMO COLOR. ¡No le pongas ropa de otro color como blanco u oscuro! El entorno, en cambio, DEBE ser de un color oscuro complementario o azul oscuro (dark blue/cyan wireframe o solid dark background) para que los personajes resalten. PROHIBIDO poner luces de otro color iluminando a los personajes (NO rim lights de otro color, NO luces naranjas iluminando a un personaje azul).
           ESTRUCTURA (HUECA / HOLLOW): Los personajes deben ser figuras de energía hechas OBLIGATORIAMENTE de mallas poligonales visibles, densas y TRANSPARENTES/HUECAS. La malla (wireframe) debe ser UNIFORME, ORDENADA, como la de un software de modelado 3D (clean perfectly spaced 3D viewport grid topology). ESTRICTAMENTE PROHIBIDO usar rellenos sólidos, bloques de color, o pintar mallas sobre piel sólida humana. No uses texturas sólidas.
           PROHIBIDO (CRÍTICO): NO incluyas NÚMEROS, NO incluyas LETRAS, NO incluyas CÓDIGO, NO incluyas NOMBRES, NO letreros de neón, ni texto de ningún tipo. La imagen debe estar totalmente limpia de letras y números. NO incluyas coches ni vehículos deportivos.
           PROHIBICIÓN ESTRICTA DE PALABRAS: BAJO NINGUNA CIRCUNSTANCIA uses las palabras "futurista", "futuristic", "sci-fi", "cyberpunk", "holograma", "neon sign" ni "car" en tus prompts o descripciones (ni en inglés ni en español). Es un mundo ACTUAL y COTIDIANO, solo que su estilo visual es un plano de malla de neón.
           Para la ropa: ¡LA ROPA DEBE SER UNA MALLA (wireframe mesh)! ESTÁ TERMINANTEMENTE PROHIBIDO usar ropa de textura sólida ("solid t-shirt", "solid fabric"). Describe prendas tecnológicas normales pero hechas de "glowing mesh" y MANTÉN SU COLOR IGUAL AL DEL CUERPO DEL PERSONAJE. Usa términos como 'glowing wireframe mesh', 'volumetric lighting', '8k'.`
        : "";

      const neonNarrativeInstruction = activeStyle === "Neon Blueprint" || activeStyle === "Neon Pro"
        ? `ESTRUCTURA NARRATIVA PARA NEON (OBLIGATORIO): 
           - TEMA: Crea historias profundas, emocionales y de gran impacto (ej. amor y traición, memoria borrada, sacrificio trágico, despertar emocional, amor prohibido, muerte digital o venganza). NUNCA repitas la misma historia; sé extremadamente creativo.
           - INICIO VIRAL: El primer segmento debe tener un 'gancho' (hook) muy impactante para atrapar a la audiencia en los primeros 3 segundos.
           - FINAL VIRAL Y PREGUNTA: El desenlace (último segmento) debe ser devastador o revelador, y terminar INVARIABLEMENTE con una pregunta profunda y emocional dirigida al espectador (ej. "¿Qué harías tú si todo fuera una mentira?").`
        : "";

      const holographicSoulInstruction = activeStyle === "HOLOGRAPHIC SOUL"
        ? `ESPECIAL PARA HOLOGRAPHIC SOUL (OBLIGATORIO Y PRIORITARIO SOBRE CUALQUIER OTRA INSTRUCCIÓN DE APARIENCIA):
           1. ESTILO BASE: Figura holográfica digital, translúcida y etérea, integrada en entornos arquitectónicos o cotidianos REALES (una sala, una calle, un espacio con luz natural) — NO en un vacío negro. Iluminación dramática, fuerte contraste entre la luz digital interna del personaje y las sombras profundas del entorno.
           2. COMPOSICIÓN LUMÍNICA: La figura brilla con luz propia y proyecta reflejos de luz volumétrica sobre el entorno real (pisos de madera, paredes, objetos cercanos). Cuerpo translúcido con detalles luminosos internos y partículas de luz flotantes o polvo estelar sutil a su alrededor. TOTALMENTE PROHIBIDO: mallas de alambre, líneas de cuadrícula técnica (wireframe), o artefactos de diseño 3D genéricos.
           ${visualAnchorImages.length > 0 ? `3. REGLA DE ORO (CONSISTENCIA DE AVATAR — HAY IMÁGENES DE REFERENCIA ADJUNTAS): Queda TOTALMENTE PROHIBIDO especificar atributos físicos concretos del personaje (longitud o color exacto de cabello, prendas de ropa predefinidas, tono de piel específico, rasgos faciales cerrados). NO inventes esos detalles. Toda descripción del personaje en 'imagePrompt' debe referirse ÚNICAMENTE, de forma abstracta, a "the ethereal holographic figure from the reference image" (o "...figures" si hay varias personas), dejando que el avatar real cargado por el usuario defina esos rasgos, sin que el texto del prompt los contradiga o los limite.` 
             : `3. REGLA DE ORO (SIN AVATAR DE REFERENCIA CARGADO): Como no hay imagen de referencia en este caso, describe a "an ethereal holographic figure" de forma DELIBERADAMENTE ABSTRACTA: evita fijar longitud/color de cabello, prendas específicas o tono de piel exacto. Mantén solo una silueta humanoide general y coherente entre segmentos (proporciones, altura relativa), sin detalles físicos cerrados que después puedan contradecir un avatar que el usuario cargue más adelante.`}
           4. CALIDAD Y TEXTURAS: Renderizado fotorrealista de alta gama (estilo 8k, fotografía arquitectónica o cinematográfica) para el ENTORNO, combinado con la figura holográfica translúcida. Luz volumétrica suave, partículas de luz flotantes, acabados limpios, libres de líneas de malla o cuadrículas.
           5. PROHIBIDO (CRÍTICO): NO incluyas números, letras, letreros ni texto de ningún tipo en la imagen. NO incluyas coches ni vehículos. NO uses las palabras "futurista", "futuristic", "sci-fi", "cyberpunk".
           Terminología recomendada: "the ethereal holographic figure from the reference image", "smooth, glowing volumetric light", "translucent body with internal luminous details", "floating light particles", "volumetric light reflections on the environment", "8k", "deep atmosphere".`
        : "";

      // Groq - motor de texto/visión para historias y prompts
      const visualAnchorNote = visualAnchorImages.length > 0
        ? (activeStyle === "HOLOGRAPHIC SOUL"
            ? `REFERENCIAS VISUALES ADJUNTAS: El usuario adjuntó ${visualAnchorImages.length} imagen(es) de referencia del/de los personaje(s). Analízalas para identificar cuántos personajes distintos hay y su silueta/pose general, pero NO describas sus rasgos físicos concretos en el texto (ver regla de oro de Holographic Soul más abajo) — solo refiérete a ellos como "the ethereal holographic figure from the reference image".`
            : `REFERENCIAS VISUALES ADJUNTAS: El usuario adjuntó ${visualAnchorImages.length} imagen(es) de referencia de personaje/estilo en este mensaje. ANALIZA CUIDADOSAMENTE cada imagen (rasgos faciales, color y estilo de pelo, tono de piel o color de energía si es un estilo holográfico/neón, vestimenta, calzado, complexión corporal) y describe ese MISMO personaje de forma EXTREMADAMENTE DETALLADA y CONSISTENTE en el campo 'imagePrompt' de TODOS los segmentos. NO inventes un personaje distinto al de las imágenes. NO contradigas lo que ves en las imágenes.`)
        : "";

      const generateWithModel = async (modelName: string) => {
        const directScriptInstruction = activePrompt.includes("NARRACIÓN") || activePrompt.includes("HOOK") 
          ? `¡ESTADO CRÍTICO! El usuario ha proporcionado un GUIÓN TEXTUAL COMPLETO. TU ÚNICA TAREA ES EXTRAER ESE TEXTO Y DIVIDIRLO EN SEGMENTOS. **ESTÁ TOTALMENTE PROHIBIDO ALTERAR, CAMBIAR, AÑADIR O QUITAR PALABRAS Y DESENLACES A LA HISTORIA.** Respeta el texto EXACTO. Asigna una voz diferente si un segmento es diálogo de otro personaje.` : ``;

        const voiceInstruction = `
          Opcional: Si en la historia hay diálogos o cambia el narrador/personaje, puedes usar el campo 'characterVoice' en el segmento para especificar quién habla.
          ${voiceProvider === 'elevenlabs' 
            ? `Las voces disponibles (elige el nombre exacto) son: ${elevenLabsVoices.map(v=>v.name).join(', ')}` 
            : `Las voces disponibles son: Aoede, Charon, Fenrir, Kore, Puck`}
        `;

        const textPrompt = `Crea un contenido de ${promptType} de CALIDAD CINEMATOGRÁFICA SUPREMA basado en: "${activePrompt}". 
          ${durationInstruction}
          ${characterIdentificationInstruction}
          ${neonBlueprintInstruction}
          ${holographicSoulInstruction}
          ${neonNarrativeInstruction}
          ${directScriptInstruction}
          ${voiceInstruction}
          

          // Configuración básica de prompts
          ${type === 'quote'
            ? "Cada segmento debe ser una frase corta y potente de máximo 20 palabras."
            : useExactText 
              ? `TEXTO INTACTO OBLIGATORIO: Está TOTALMENTE PROHIBIDO modificar la historia, guión o resumen proporcionado por el usuario. No puedes agregar palabras, no puedes omitir partes, ni reescribir con tus palabras. Tienes que DIVIDIR literalmente el texto exacto proporcionado en segmentos cortos. Debes devolver la narración exactamente igual que la recibiste en el campo 'narration' del JSON de salida.` 
              : isReflectionMode ? "Cada segmento debe ser una reflexión profunda de máximo 20 palabras." : "Crea una narrativa rica dividida en segmentos cortos de máximo 20 palabras. IMPORTANTE: Para cumplir con la duración, debes crear entre 10 y 11 segmentos por cada minuto de narración."}
          Para cada segmento, proporciona el texto, una descripción visual detallada (EN ESPAÑOL), un ángulo de cámara y una DESCRIPCIÓN DE MOVIMIENTO Y ESCENA (EN ESPAÑOL).
          
          REGLAS PARA 'imagePrompt' y 'videoDescription' (DESCRIPCIONES - EN ESPAÑOL):
          1. REDACCIÓN ESTRICTA: La descripción DEBE describir SOLAMENTE la acción, el entorno (actual, presente, natural, rústico o doméstico) y la ropa de los personajes, SIN MENCIONAR COLOR, SIN DETALLES FÍSICOS O ANATÓMICOS, SIN FUTURISMO.
          2. ¡SUPER DETALLADO! Este campo NO puede ser corto y perezoso. Debe ser una descripción profunda y rica.
          3. Describe el tipo de movimiento exacto o la acción de forma fluida y cinemática (caminar, gesticular, parpadear).
          4. Describe la ambientación y los movimientos de cámara (paneos, zooms, iluminación ambiental) en tiempo actual.
          5. AL FINAL DE CADA DESCRIPCIÓN DE MOVIMIENTO, añade SIEMPRE EXACTAMENTE esta frase: "No poner música ni audio de voz."
          
          ${activeStyle === "Dios" ? `
          ESTRUCTURA PARA ESTILO DIOS:
          1. Gancho impactante.
          2. Cuerpo solemne y compasivo.
          3. Llamado a la acción (CTA).

          ` : ""}
          ${reflectionInstruction}
          ${dynamicAnglesInstruction}
          ${!isReflectionMode ? `
          IMPORTANTE: Personajes de BELLEZA SUPREMA, atractivos y consistentes. Define un color de pelo y mantenlo.` : "Variedad visual total entre segmentos."}
          ${visualAnchorNote}
          Asegúrate de que cada segmento sea visualmente espectacular, artístico y de calidad suprema.

          FORMATO DE SALIDA (OBLIGATORIO, SIN EXCEPCIONES): Responde ÚNICAMENTE con un objeto JSON válido, sin texto adicional antes ni después, sin markdown, sin envolver el JSON en otra clave. La clave raíz "segments" es OBLIGATORIA y debe ser un array con AL MENOS un elemento, con EXACTAMENTE esta forma (no anides este objeto dentro de otra clave como "story" o "data", no lo devuelvas como array suelto):
          {
            "title": string,
            "narration": string,
            "hashtags": string[],
            "segments": [
              {
                "text": string,
                "imagePrompt": string, // EN INGLÉS, MÍNIMO 150 PALABRAS, extremadamente detallado (iluminación, atmósfera, cámara). Mantén al personaje consistente.
                "videoDescription": string, // EN ESPAÑOL, describe movimiento y vestimenta, termina siempre con "No poner música ni audio de voz."
                "cameraAngle": string,
                "characterVoice": string // opcional
              }
            ]
          }`;

        // Solo usamos el formato de "content" en arreglo (partes de texto/imagen)
        // cuando realmente hay imágenes que mandar — el modelo de texto puro
        // (gpt-oss-120b) rechaza ese formato con un 400 si le llega un arreglo,
        // aunque solo tenga una parte de texto. Con imágenes sí es necesario
        // porque así es como el modelo de visión (qwen) recibe cada imagen.
        const userContent: string | GroqContentPart[] = visualAnchorImages.length > 0
          ? [
              { type: "text", text: textPrompt },
              // qwen3.6-27b admite hasta 5 imágenes por request
              ...visualAnchorImages.slice(0, 5).map(img => ({ type: "image_url" as const, image_url: { url: img } }))
            ]
          : textPrompt;

        const groqResponse = await withRetry(() => callGroq([
          { role: "system", content: "Eres un guionista y director de arte experto en contenido viral para redes sociales. SIEMPRE respondes con JSON válido y nada más, sin explicaciones ni markdown." },
          { role: "user", content: userContent }
        ], modelName));

        return groqResponse;
      };

      const hasReferenceImages = visualAnchorImages.length > 0;
      let rawText: string;
      try {
        rawText = await generateWithModel(hasReferenceImages ? "qwen/qwen3.6-27b" : "openai/gpt-oss-120b");
      } catch (err: any) {
        console.warn(`Fallo con el modelo principal, probando respaldo...`, err);
        try {
          rawText = await generateWithModel(hasReferenceImages ? "qwen/qwen3.8-27b" : "openai/gpt-oss-20b");
        } catch (err2: any) {
          if (hasReferenceImages) {
            // Último recurso: seguir sin imágenes en vez de fallar del todo.
            console.warn("Fallaron ambos modelos de visión, generando sin analizar las imágenes...", err2);
            rawText = await generateWithModel("openai/gpt-oss-120b");
          } else {
            throw err2;
          }
        }
      }

      if (!rawText) {
        throw new Error("Groq no devolvió ninguna respuesta.");
      }

      const data = parseGroqJson(rawText);

      // Groq no fuerza un schema estricto como Gemini, así que a veces el
      // array de segmentos viene anidado bajo una clave distinta o el
      // modelo devuelve directamente un array. Intentamos varias formas
      // razonables antes de fallar con un mensaje claro.
      let rawSegments: any = data?.segments;
      if (!Array.isArray(rawSegments)) {
        if (Array.isArray(data)) {
          rawSegments = data;
        } else if (Array.isArray(data?.story?.segments)) {
          rawSegments = data.story.segments;
        } else if (Array.isArray(data?.data?.segments)) {
          rawSegments = data.data.segments;
        } else {
          const firstArrayValue = data && typeof data === 'object'
            ? Object.values(data).find((v: any) => Array.isArray(v) && v.length > 0 && typeof v[0] === 'object')
            : null;
          rawSegments = firstArrayValue || null;
        }
      }
      if (!Array.isArray(rawSegments) || rawSegments.length === 0) {
        console.error("Respuesta de Groq sin 'segments' válido:", data);
        throw new Error("Groq devolvió una respuesta con un formato inesperado (sin 'segments'). Intenta generar de nuevo; si persiste, prueba con una duración/número de segmentos menor.");
      }

      addCost(COSTS.STORY_GEN, 'stories');
      const formattedSegments = rawSegments.map((s: any, i: number) => ({
        ...s,
        text: s.text || `Segmento ${i + 1} de la historia.`,
        id: `seg-${i}-${Date.now()}`
      }));
      
      const fullNarration = data.narration || formattedSegments.map((s: any) => s.text).join(' ');

      const newStory: Story = { 
        id: `story-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        title: data.title || "Historia sin título", 
        narration: fullNarration, 
        hashtags: data.hashtags || [], 
        segments: formattedSegments,
        type: type,
        createdAt: Date.now()
      };

      setStory(newStory);
      setCurrentStoryId(newStory.id!); // Update currentStoryId in settings too
      storyRef.current = newStory; 
      setVisualAnchor(null); 
      setActiveSegmentIndex(0);
      
      // Also save to Firestore immediately if user is logged in
      if (user) {
        const storyRef = doc(db, 'users', user.uid, 'stories', newStory.id!);
        const cleanStory = sanitizeForFirestore({
          ...newStory,
          userId: user.uid,
          updatedAt: Date.now()
        });
        setDoc(storyRef, cleanStory, { merge: true }).catch(err => {
          handleFirestoreError(err, OperationType.WRITE, `users/${user.uid}/stories/${newStory.id}`);
        });
      }

      return newStory;
    } catch (err: any) {
      console.error("Story generation error:", err);
      let errorMessage = "Error al generar la historia. ";
      
      const errorStr = JSON.stringify(err).toLowerCase();
      if (errorStr.includes("504") || errorStr.includes("timeout")) {
        errorMessage = "La solicitud tardó demasiado (Error 504). Por favor, intenta con un prompt más corto o inténtalo de nuevo.";
      } else if (errorStr.includes("503") || errorStr.includes("high demand") || errorStr.includes("unavailable")) {
        errorMessage = "Los servidores de Google están saturados (Error 503). Hemos intentado reintentar automáticamente, pero la demanda sigue siendo muy alta. Por favor, espera un minuto e inténtalo de nuevo.";
      } else if (errorStr.includes("502") || errorStr.includes("bad gateway")) {
        errorMessage = "Error de conexión temporal (Error 502). Los servidores están saturados, por favor intenta de nuevo en unos segundos.";
      } else if (errorStr.includes("500") || errorStr.includes("internal error")) {
        errorMessage = "Error interno de Google (Error 500). Los servidores están experimentando un fallo temporal. Por favor, reintenta en unos segundos.";
      } else if (err.message?.includes("API Key")) {
        errorMessage = "Error de API Key. Por favor, selecciona una clave válida.";
      } else if (err.message?.includes("safety")) {
        errorMessage = "El contenido fue bloqueado por los filtros de seguridad. Intenta con un prompt más suave.";
      } else if (err.message?.includes("JSON")) {
        errorMessage = "Error al procesar la respuesta. Inténtalo de nuevo.";
      } else {
        errorMessage += err.message || "Por favor, inténtalo de nuevo.";
      }
      setError(errorMessage);
      return null;
    } finally {
      setIsGeneratingStory(false);
    }
  };

  const handleQuickTheme = async (theme: typeof QUICK_THEMES[0]) => {
    setPrompt(theme.prompt);
    setVisualStyle(theme.visualStyle);
    const newStory = await generateStory(theme.prompt, theme.visualStyle, 'quote');
    if (newStory) {
      generateAllImages(newStory);
    }
  };

  const overlayTextOnImage = (base64Image: string, text: string): Promise<string> => {
    return new Promise((resolve) => {
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.onload = () => {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        if (!ctx) {
          resolve(base64Image);
          return;
        }

        canvas.width = img.width;
        canvas.height = img.height;

        // Draw original image
        ctx.drawImage(img, 0, 0);

        // Text settings
        const padding = canvas.width * 0.15;
        const maxWidth = canvas.width - (padding * 2);
        const fontSize = Math.max(24, canvas.width * 0.04);
        ctx.font = `italic ${fontSize}px "Playfair Display", serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';

        // Split text into lines with better wrapping
        const words = text.split(' ');
        const lines = [];
        let currentLine = '';

        for (let i = 0; i < words.length; i++) {
          const word = words[i];
          const testLine = currentLine ? currentLine + " " + word : word;
          const metrics = ctx.measureText(testLine);
          const testWidth = metrics.width;
          
          if (testWidth < maxWidth) {
            currentLine = testLine;
          } else {
            if (currentLine) lines.push(currentLine);
            currentLine = word;
          }
        }
        if (currentLine) lines.push(currentLine);

        // Calculate layout
        const lineHeight = fontSize * 1.3;
        const totalHeight = lines.length * lineHeight;
        const startY = (canvas.height / 2) - (totalHeight / 2);

        // Draw semi-transparent background box
        ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
        const boxPadding = 40;
        const boxWidth = canvas.width - (padding * 2) + (boxPadding * 2);
        const boxHeight = totalHeight + (boxPadding * 2);
        
        // Rounded rectangle for the box
        const x = padding - boxPadding;
        const y = startY - boxPadding;
        const radius = 20;
        
        ctx.beginPath();
        ctx.moveTo(x + radius, y);
        ctx.lineTo(x + boxWidth - radius, y);
        ctx.quadraticCurveTo(x + boxWidth, y, x + boxWidth, y + radius);
        ctx.lineTo(x + boxWidth, y + boxHeight - radius);
        ctx.quadraticCurveTo(x + boxWidth, y + boxHeight, x + boxWidth - radius, y + boxHeight);
        ctx.lineTo(x + radius, y + boxHeight);
        ctx.quadraticCurveTo(x, y + boxHeight, x, y + boxHeight - radius);
        ctx.lineTo(x, y + radius);
        ctx.quadraticCurveTo(x, y, x + radius, y);
        ctx.closePath();
        ctx.fill();

        // Draw text
        ctx.fillStyle = 'white';
        ctx.shadowColor = 'rgba(0, 0, 0, 0.5)';
        ctx.shadowBlur = 4;
        ctx.shadowOffsetX = 2;
        ctx.shadowOffsetY = 2;

        lines.forEach((line, index) => {
          ctx.fillText(line, canvas.width / 2, startY + (index * lineHeight) + (lineHeight / 2));
        });

        resolve(canvas.toDataURL('image/png'));
      };
      img.onerror = () => resolve(base64Image);
      img.src = base64Image;
    });
  };

  const saveCurrentCharacter = (name: string) => {
    if (!characterDesc) return;
    const newChar = {
      id: Date.now().toString(),
      name,
      description: characterDesc,
      style: visualStyle,
      imageUrl: visualAnchorImages[0] || undefined
    };
    setSavedCharacters(prev => [...prev, newChar]);
  };

  const deleteCharacter = (id: string) => {
    setSavedCharacters(prev => prev.filter(c => c.id !== id));
  };

  // Learning Logic: Analyze the first image to extract a better visual anchor
  const learnFromImage = async (imageUrl: string) => {
    if (!imageUrl) return;
    try {
      const apiKey = getApiKey();
      if (!apiKey) return;
      const ai = new GoogleGenAI({ apiKey });
      
      const response = await ai.models.generateContent({
        model: "gemini-3-flash-preview",
        contents: [
          {
            parts: [
              { text: "Describe this character's visual features in extreme detail for an image generation prompt. Focus on: SUPREME ATTRACTIVENESS, perfect facial structure, hair style and color, specific clothing details, and unique markings. Ensure the description captures the VIBRANT COLORS and high-quality aesthetic. Be concise but precise. Output ONLY the description." },
              { inlineData: { data: imageUrl.split(',')[1], mimeType: "image/png" } }
            ]
          }
        ]
      });

      const refinedDescription = response.text;
      if (refinedDescription) {
        setVisualAnchor(refinedDescription);
        console.log("App learned from image:", refinedDescription);
      }
    } catch (error) {
      console.error("Error learning from image:", error);
    }
  };

  const generateImage = async (index: number) => {
    // 1. Set loading state
    setStory(prev => {
      if (!prev) return prev;
      const updatedSegments = [...prev.segments];
      updatedSegments[index] = { ...updatedSegments[index], isGeneratingImage: true, error: undefined };
      return { ...prev, segments: updatedSegments };
    });

    try {
      const validationError = validateApiKey('visual');
      if (validationError) {
        throw new Error(validationError);
      }
      
      const currentStory = storyRef.current;
      const segment = currentStory?.segments[index];
      if (!segment) {
        setStory(prev => {
          if (!prev) return prev;
          const updatedSegments = [...prev.segments];
          if (updatedSegments[index]) {
            updatedSegments[index] = { ...updatedSegments[index], isGeneratingImage: false };
          }
          return { ...prev, segments: updatedSegments };
        });
        return;
      }

      const isReflectionMode = storyDuration.includes("imgs") || visualStyle.includes("Reflection");

      const angleInstruction = segment.cameraAngle === "Close-up" ? "Close-up shot focusing on the character's face and emotions, shallow depth of field." :
                              segment.cameraAngle === "Extreme Close-up" ? "Extreme close-up macro shot, focus on a specific detail like eyes or an object, high detail." :
                              segment.cameraAngle === "Conversation" ? "Medium shot, cinematic depth of field, focus on the character's interaction with the environment." :
                              segment.cameraAngle === "Aerial" ? "High angle aerial shot, bird's eye view, showing the scale of the scene from above." :
                              segment.cameraAngle === "Wide Shot" ? "Wide panoramic shot, establishing the environment and the character's position within it." :
                              segment.cameraAngle === "Fisheye" ? "Extreme wide angle fisheye lens distortion, immersive and dynamic perspective." :
                              segment.cameraAngle === "Low Angle" ? "Low angle shot looking up at the character, making them look powerful and imposing." :
                              segment.cameraAngle === "Dutch Angle" ? "Dutch angle shot, tilted camera frame, creating a sense of dynamic action or tension." :
                              segment.cameraAngle === "Mirror Reflection" ? "A shot of the character's reflection in a mirror, window, or water surface, artistic composition." :
                              "Standard cinematic camera angle.";

      // Visual Style Instruction
      const styleInstruction = visualStyle === "Pixar Anime" ? "High-end 3D animation, Pixar and Ghibli hybrid style, Octane Render, 8k, vibrant colors" : 
                             visualStyle === "Reflection Dark" ? "Dark, moody, high contrast, cinematic shadows, minimalist" :
                             visualStyle === "Reflection Nature" ? "Breathtaking nature landscapes, cinematic lighting, wide angle" :
                             visualStyle === "Reflection Urban" ? "Urban cityscapes, rainy streets, neon highlights, modern architecture" :
                             visualStyle === "Gold Skeleton" ? "Solid golden human skeleton, shiny, hyper-realistic, 8k, cinematic lighting, liquid gold textures, reflective surfaces" :
                             visualStyle === "Glass Skeleton" ? "Translucent glass human skeleton with visible internal organs, glowing core, hyper-realistic, 8k, cinematic lighting, glass and gold textures, refractive surfaces" :
                             visualStyle === "Cyber-Schematic" ? "Cyber-Schematic Wireframe, Holographic Combat HUD, technical wireframe mesh, high contrast Amber and Cyan, HUD interface elements (Critical Hit, energy bars), electric sparks, grid distortions, futuristic tactical simulation" :
                             visualStyle === "Manhwa Premium" ? "Masterpiece, Supreme Quality, Ultra detailed premium mature manhwa style, semi-realistic digital illustration, perfect airbrushed glowing skin, symmetrical hyper-idealized features, cinematic rim lighting, vibrant saturated colors. FEMALE CHARACTER: supreme beauty, extremely attractive, hyper-idealized exaggerated hourglass figure, large lifted bust, narrow waist, wide hips, thick thighs, sculpted athletic body, long flowing silky hair, alluring and seductive expression. MALE CHARACTER: supreme handsomeness, massive build, wide shoulders, broad chest, large pectorals, alpha male physique, sharp jawline, intense gaze, rugged handsomeness." :
                              visualStyle === "Neon Blueprint" ? "Director Persona: You are the visual director of the NEON BLUEPRINT universe. Every scene is a 3D technical blueprint built from intense glowing wireframes. Clothing must be distinct per scene." :
                             visualStyle === "Neon Pro" ? "Director Persona: You are the visual director of the NEON BLUEPRINT universe. High-contrast futuristic urban environment built from glowing wireframes. Clothing MUST change per scene." :
                             visualStyle === "HOLOGRAPHIC SOUL" ? "Director Persona: You are the visual director of the HOLOGRAPHIC SOUL universe. The subject must be a PURELY ABSTRACT, GAS-LIKE, VOLUMETRIC CLOUD OF LUMINOUS ENERGY. ABSOLUTELY NO HUMAN FEATURES: No face, no eyes, no hair, no skin, no limbs, no clothes, no solid surfaces. It is NOT a silhouette of a person, it is NOT a human shape, it is NOT a mannequin. It must be completely shapeless, like swirling nebula gas or ethereal aurora energy. Cinematic dark background with bokeh highlights. MANDATORY: NO 'human', NO 'person', NO 'face', NO 'skin', NO 'anatomy', NO 'hair', NO 'body', NO 'shape', NO 'silhouette'." :
                             visualStyle === "3D Animation" ? "Masterpiece, High-end 3D Animation style, Pixar or Disney aesthetic, extremely high quality, expressive characters with large eyes and animated proportions, detailed but stylized textures, vibrant lighting, soft rendering, 8k, cinematic camera, colorful and lively atmosphere, smooth gradients." :
                             visualStyle === "Power Couple" ? "Masterpiece, Supreme Quality, High-end digital illustration, luxury mature comic style, premium graphic novel aesthetic. Elegant strokes, dramatic volumetric shading, high-contrast lighting. CHARACTERS: Stunningly beautiful, supreme attractiveness, hyper-idealized exaggerated hourglass physiques, seductive poses, vibrant colors, sophisticated atmosphere." :
                             visualStyle === "Comic Realista Moderno" ? "Masterpiece, Supreme Quality, High-end Realistic Graphic Novel style, clean sharp black outlines, dramatic chiaroscuro lighting, high-contrast. CHARACTERS: Hyper-idealized and sculptural anatomy, extremely attractive and seductive. FEMALE: Hyper-idealized exaggerated hourglass figure, large lifted bust, narrow waist, wide hips, thick thighs, sculpted athletic body, long flowing silky hair, alluring and seductive expression. MALE: Massive build, wide shoulders, broad chest, large pectorals, alpha male physique, sharp jawline, intense gaze. SCENES: Cinematic composition, deep shadows, sophisticated color palette, high-definition textures, 8k, octane render style." :
                             visualStyle === "Muscular Fitness" ? "Ultra detailed anime male bodybuilder turnaround sheet, bald muscular man, massive heroic proportions, huge chest, giant shoulders, ultra defined 8-pack abs, gigantic arms and legs, shiny skin highlights, serious face, black tight shorts, EXACTLY 3 views (side, front, back), full body visible, dark industrial background with red cinematic lighting and smoke, premium anime rendering, AAA character design, clean lineart, soft cel shading, masterpiece, hyper detailed anatomy, symmetrical character sheet, NO text, NO labels, NO watermarks, NO letters, NO words." :
                             (visualStyle as string) === "Ancient Prophet" || visualStyle === "Historias de Dios" ? "Photorealistic, cinematic historical epic style, ancient Middle Eastern desert landscape, dramatic lighting, weathered textures, dust particles in the air, intense expressive face, detailed clothing, soft natural sunlight, masterpiece, 8k" :
                             visualStyle;

      const consistencyInstruction = consistencyLevel === "Extreme" 
        ? "STRICT ADHERENCE: You MUST match the reference image EXACTLY. Do not deviate from the character's facial features, hair, or clothing. This is a high-stakes consistency task."
        : "Maintain visual consistency with the character and style.";

      // Learning Logic: If we have a visual anchor, use it to enforce consistency
      const previousSegment = index > 0 ? currentStory?.segments[index - 1] : null;
      const contextInstruction = previousSegment 
        ? `PREVIOUS SCENE CONTEXT: The previous scene showed: "${previousSegment.imagePrompt}". 
           CONTINUITY: Ensure the current scene flows naturally from the previous one. Maintain the same lighting, time of day, and environmental details unless a transition is explicitly described.`
        : "";

      const finalPrompt = visualAnchor 
        ? `BASE CHARACTER IDENTITY: ${visualAnchor}. 
           ${contextInstruction}
           CURRENT SCENE ACTION: ${segment.imagePrompt}. 
           CAMERA ANGLE & PERSPECTIVE: ${angleInstruction}.
           VISUAL STYLE: ${styleInstruction}.
           ${consistencyInstruction}`
        : `MASTER CHARACTER IDENTITY: ${characterDesc || "Unique character design defined in the story segments"}. 
           ${contextInstruction}
           SCENE DESCRIPTION: ${segment.imagePrompt}. 
           CAMERA ANGLE & PERSPECTIVE: ${angleInstruction}.
           VISUAL STYLE: ${styleInstruction}.
           ${consistencyInstruction}
           IMPORTANT: Maintain visual consistency within this story but ensure unique character and environment details.`;

      const fullPrompt = `${finalPrompt}
          ${!visualAnchor && visualStyle !== "Muscular Fitness" && visualStyle !== "HOLOGRAPHIC SOUL" ? `REQUIREMENTS: Extremely attractive characters, SUPREME BEAUTY, perfect symmetrical faces, hyper-idealized anatomy. 
          VIBRANT COLOR PALETTE: Use a rich, saturated, and artistic scene with dramatic lighting.
          FEMALE CHARACTERS: supreme beauty, hyper-idealized exaggerated hourglass figure, large lifted bust, narrow waist, wide hips, thick thighs, sculpted athletic body, seductive and alluring.
          MALE CHARACTERS: supreme handsomeness, massive build, wide shoulders, broad chest, large pectorals, alpha male presence, sharp jawline, intense gaze.
          Consistent facial features, STRICTLY CONSISTENT HAIR COLOR AND STYLE.` : ""}
          ${visualStyle.includes("Neon") || visualStyle === "HOLOGRAPHIC SOUL" ? 
            `PROMPT PERSONA & RULES: You are the visual director of this cinematic universe. 
             Mandatory: Full scenes with characters AND detailed 3D/urban environments. NO empty voids.
             Lighting: Dramatic, high contrast, vibrant neon glow. 
             Characters: Athletic/Guitar-body physiques, fully clothed per scene, NO NUDITY, NO BAREFOOT, NO BALDNESS.
             Clothing: MUST change completely in EVERY scene. 
             Style: ${visualStyle === "HOLOGRAPHIC SOUL" ? "Pure Volumetric Ethereal Light Energy only." : "Pure Glowing Wireframe/Holographic energy only."} NO SOLID FILLS.` : 
            "CLOTHING CONSISTENCY: The characters MUST wear the EXACT same outfit as in the reference or previous segments unless a change is explicitly mentioned."}
          STRICTLY NO TEXT, NO SPEECH BUBBLES, NO DIALOGUE, NO WATERMARKS.`;

      // Select model based on quality setting and key availability
      const apiKey = getApiKey();
      const isPro = modelQuality === "Pro";
      const hasKey = !!apiKey;
      
      let imageUrl = '';
      let cost = 0;

      if (imageProvider === "Pollinations") {
        // Pollinations.ai (Free) - Optimized for stability
        const width = imageSize === "1K" ? 1024 : imageSize === "2K" ? 2048 : 512;
        const height = videoAspectRatio === "16:9" ? Math.round(width * 9 / 16) : 
                       videoAspectRatio === "9:16" ? Math.round(width * 16 / 9) : width;
        
        const seed = Math.floor(Math.random() * 1000000);
        
        // Optimized prompt for Pollinations: Style first to prevent truncation, increased limit
        const promptBase = segment.imagePrompt || "beautiful cinematic scene";
        const qualityBoost = isPro ? "photorealistic, ultra-realistic, 8k, highly detailed skin textures, masterpiece, cinematic lighting, supreme attractiveness, perfect faces" : "highly detailed, sharp focus";
        const requirements = !visualAnchor ? `Requirements: Extremely attractive characters, perfect symmetrical faces, athletic physiques, high contrast. ` : "";
        
        const styleBoost = (visualStyle === "Muscular Fitness" || visualStyle === "HOLOGRAPHIC SOUL") ? "" : `${requirements}${qualityBoost}`;

        // Strict consistency protocol: Force adherence to reference character
        const femaleChar = "Mujer";
        const maleChar = "Hombre";
        
        const isFemale = /woman|her|she|female|mujer|ella/i.test(promptBase);
        const isMale = /man|his|he|male|hombre|él/i.test(promptBase);
        const subject = (isFemale && isMale) ? "El hombre y la mujer" : (isFemale ? "La mujer" : "El hombre");
        const charIndicator = (isFemale && isMale) ? "🔴🔵" : (isFemale ? "🔵" : "🔴");

        // Sanitize input: Remove forbidden phrases (Aggressive filtering)
        const forbiddenPhrases = [
            // Colors
            /azul/gi, /rojo/gi, /verde/gi, /morado/gi, /violeta/gi, /púrpura/gi, /cian/gi, /rosa/gi, /negro/gi, /blanco/gi, /gris/gi, /amarillo/gi, /naranja/gi, /plateado/gi, /dorado/gi,
            // Tech/Futuristic
            /holográfico/gi, /holográfica/gi, /holograma/gi, /futurista/gi, /neón/gi, /cyberpunk/gi, /digital/gi, /robot/gi, /androide/gi, /tecnología/gi, /electrónico/gi, /pantalla/gi, /líneas de escaneo/gi, /neuronal/gi, /futuro/gi, /ciudad/gi, /edificios/gi, /data/gi, /bits/gi, /código/gi, /binario/gi, /cifras/gi, /números/gi,
            // Appearance/Body/Clothing (Aggressive list)
            /reloj de arena/gi, /cuerpo/gi, /físico/gi, /busto/gi, /senos/gi, /silueta/gi, /anatómico/gi, /atractivo/gi, /musculoso/gi, /atlético/gi, /abs/gi, /facial/gi, /piel/gi, /poros/gi, /textura/gi, /largo/gi, /corto/gi, /grande/gi, /pequeño/gi, /perfecto/gi, /estupendo/gi, /hermoso/gi, /bello/gi, /impresionante/gi, /hombre/gi, /mujer/gi, /persona/gi, /humano/gi, /rostro/gi, /cara/gi, /ojos/gi, /labios/gi, /cabello/gi, /pelo/gi, /guapo/gi, /realista/gi, /photorealistic/gi, /fotorrealista/gi, /contemporáneo/gi, /moderno/gi, /lujoso/gi, /suave/gi, /intenso/gi, /brillante/gi, /oscuro/gi, /claro/gi, /muebles/gi, /decoración/gi, /ventana/gi, /vidrio/gi, /pared/gi, /suelo/gi, /cielo/gi, /atardecer/gi, /amanecer/gi, /lluvia/gi,
            /camisón|bata|ropa|tela|falda|blusa|vestido|pantalones|jersey|cuello alto|camisa|camiseta|shorts|minifalda|tacones|calzado|zapatos|raso|encaje|estilo|braids|trenzas|mahogany|caoba|oro|gold/gi,
            // Ethereal/Abstract
            /fantasma/gi, /fantasmagórico/gi, /fantasmagórica/gi, /espíritu/gi, /aura/gi, /brillante/gi, /transparente/gi, /translucido/gi, /faded/gi, /desvanecido/gi, /energía/gi, /luz/gi, /mágico/gi, /irreal/gi, /melancolía/gi, /triste/gi, /nostalgia/gi, /dolor/gi
        ];
        let sanitizedPrompt = promptBase;
        forbiddenPhrases.forEach(regex => {
            sanitizedPrompt = sanitizedPrompt.replace(regex, "");
        });

        const actionPart = `${subject} ${sanitizedPrompt.replace(/\s+/g, ' ').trim()}`;
        
        let finalPromptBase;
        if (visualStyle === "HOLOGRAPHIC SOUL") {
            finalPromptBase = `Ethereal volumetric light energy hologram style, ethereal, abstract, wireframe. Subject: ${subject}. Action: ${sanitizedPrompt.replace(/\s+/g, ' ').trim()}. STRICT RULE: NO DESCRIPTION OF CLOTHING, BODY, BACKGROUND, OR LIGHTING.`;
        } else if (visualStyle.includes("Neon")) {
            finalPromptBase = `Vibrant neon glowing wireframe style, digital aesthetic. ${actionPart}.`;
        } else if (visualStyle === "Minimal Clay Animation" || (visualStyle as string) === "Clay Animation") {
            finalPromptBase = `Minimal clay animation style, stop-motion aesthetic, matte clay texture, earthy color palette, visible clay imperfections. ${actionPart}.`;
        } else {
            finalPromptBase = `${actionPart}. ${styleBoost}`;
        }
        
        const simplePrompt = `Description: ${finalPromptBase}`.substring(0, 2000);
        
        console.log("Generating with Pollinations:", simplePrompt);
        
        const tryPollinations = async (model: string | null) => {
          const modelParam = model ? `&model=${model}` : "";
          const pollUrl = `https://image.pollinations.ai/prompt/${encodeURIComponent(simplePrompt)}?width=${width}&height=${height}&seed=${seed}${modelParam}&nologo=true`;
          
          const resp = await fetch(pollUrl);
          if (!resp.ok) throw new Error(`Model ${model || "default"} failed`);
          const blob = await resp.blob();
          return new Promise<string>((resolve, reject) => {
            const reader = new FileReader();
            reader.onloadend = () => resolve(reader.result as string);
            reader.onerror = reject;
            reader.readAsDataURL(blob);
          });
        };

        // Try models based on quality setting
        try {
          if (isPro) {
            // Try Flux (Best quality)
            imageUrl = await tryPollinations("flux");
          } else {
            // Try Turbo (Fastest/Most stable)
            imageUrl = await tryPollinations("turbo");
          }
        } catch (err) {
          console.warn(`Pollinations ${isPro ? "Flux" : "Turbo"} failed, trying fallback...`, err);
          try {
            // Fallback to the other one
            imageUrl = await tryPollinations(isPro ? "turbo" : "flux");
          } catch (err2) {
            console.warn("Pollinations fallback failed, trying Default...", err2);
            try {
              // Try Default (No model specified)
              imageUrl = await tryPollinations(null);
            } catch (err3) {
              // FINAL FALLBACK: If Pollinations is completely down, try Google AI automatically
              console.error("Pollinations completely failed, falling back to Google AI...");
              const apiKey = getApiKey();
              if (!apiKey) throw new Error("Pollinations API unavailable and no Gemini Key configured for fallback.");
              
              const ai = new GoogleGenAI({ apiKey });
              const modelName = 'gemini-2.5-flash-image';
              const parts: any[] = [{ text: fullPrompt }];
              const response = await withRetry(() => ai.models.generateContent({
                model: modelName,
                contents: { parts },
                config: { imageConfig: { aspectRatio: videoAspectRatio, imageSize: "1K" } },
              }));
              for (const part of response.candidates?.[0]?.content?.parts || []) {
                if (part.inlineData) {
                  imageUrl = `data:image/png;base64,${part.inlineData.data}`;
                  break;
                }
              }
              if (!imageUrl) throw new Error("Pollinations API unavailable and Google fallback failed.");
            }
          }
        }
        cost = 0; // Free tier or fallback
      } else {
        // Google GenAI (Flash Tier)
        const apiKey = getApiKey();
        if (!apiKey) throw new Error("Por favor, configura tu API Key de Gemini para usar el proveedor de Google.");
        const ai = new GoogleGenAI({ apiKey });

        // Use 3.1 Flash as primary for Pro quality, 2.5 as fallback
        let modelToUse = isPro ? 'gemini-3.1-flash-image-preview' : 'gemini-2.5-flash-image';
        cost = 0; 

        // Prepare parts for multimodal consistency - Increased limit to 3000 to avoid truncation
        const parts: any[] = [{ text: fullPrompt.substring(0, 3000) }]; 
        if (visualAnchorImages.length > 0) {
          visualAnchorImages.forEach(img => {
            parts.unshift({
              inlineData: {
                data: img.split(',')[1],
                mimeType: "image/png"
              }
            });
          });
        }

        const executeGeneration = async (model: string) => {
          // Increased retry delay to 10s specifically for images to clear rate limits
          return await withRetry(() => ai.models.generateContent({
            model: model,
            contents: {
              parts: parts,
            },
            config: {
              imageConfig: { 
                aspectRatio: videoAspectRatio, 
                imageSize: isPro ? imageSize : "1K" 
              }
            },
          }), 3, 10000); // 3 retries, 10s delay
        };

        let response;
        try {
          response = await executeGeneration(modelToUse);
        } catch (err: any) {
          const errStr = String(err.message || "").toLowerCase();
          const isInternal = errStr.includes("internal") || errStr.includes("500");
          const isSafety = errStr.includes("safety") || errStr.includes("blocked");
          
          if (isSafety) {
            throw new Error("Google bloqueó la imagen por sus filtros de seguridad. Intenta cambiar el estilo a uno menos provocativo o usa Pollinations.");
          }

          // Fallback logic: If 3.1 fails, try 2.5
          if (isInternal && modelToUse === 'gemini-3.1-flash-image-preview') {
            console.warn("Switching to gemini-2.5-flash-image due to internal error in 3.1");
            modelToUse = 'gemini-2.5-flash-image';
            response = await executeGeneration(modelToUse);
          } else if (errStr.includes("failed to fetch")) {
            throw new Error("Error de conexión: No se pudo contactar con el servidor. Revisa tu internet o desactiva extensiones que bloqueen anuncios.");
          } else {
            throw err;
          }
        }

        for (const part of response.candidates?.[0]?.content?.parts || []) {
          if (part.inlineData) {
            imageUrl = `data:image/png;base64,${part.inlineData.data}`;
            break;
          }
        }
      }

      if (imageUrl) {
        // Overlay text on the image if enabled (ONLY for quotes)
        const finalImageUrl = (burnTextIntoImages && currentStory?.type === 'quote') 
          ? await overlayTextOnImage(imageUrl, segment.text)
          : imageUrl;

        // If this is the first image and no anchors exist, set it as the first visual anchor
        if (visualAnchorImages.length === 0) {
          setVisualAnchorImages([finalImageUrl]);
          // Trigger learning logic
          learnFromImage(finalImageUrl);
        }
        
        addCost(cost, 'images');
        setStory(prev => {
          if (!prev) return prev;
          const finalSegments = [...prev.segments];
          finalSegments[index] = { ...finalSegments[index], imageUrl: finalImageUrl, isGeneratingImage: false };
          return { ...prev, segments: finalSegments };
        });
      } else {
        throw new Error("No image data received from model");
      }
    } catch (err: any) {
      console.error("Image generation error:", err);
      const errorStr = JSON.stringify(err).toLowerCase();
      const isSafetyError = errorStr.includes("safety") || 
                           err.message?.toLowerCase().includes("safety");
      const isQuotaError = errorStr.includes("quota") || 
                          err.message?.toLowerCase().includes("quota") ||
                          err.message?.toLowerCase().includes("429");
      const isInternalError = errorStr.includes("internal error") || 
                             errorStr.includes("500") ||
                             err.message?.toLowerCase().includes("internal");
      const isFetchError = errorStr.includes("failed to fetch") || 
                          err.message?.toLowerCase().includes("fetch") ||
                          err.message?.toLowerCase().includes("conexión");
      
      setStory(prev => {
        if (!prev) return prev;
        const finalSegments = [...prev.segments];
        finalSegments[index] = { 
          ...finalSegments[index], 
          isGeneratingImage: false, 
          error: isSafetyError 
            ? "Contenido bloqueado (Filtro de seguridad). Intenta un prompt más suave." 
            : isQuotaError
            ? "Límite de velocidad alcanzado. Espera un momento y reintenta."
            : isInternalError
            ? "Error interno de Google (Servidor saturado). Reintenta en unos segundos o cambia el estilo visual."
            : isFetchError
            ? (err.message || "Error de red. Revisa tu conexión a internet.")
            : (err.message || "Error al generar imagen.")
        };
        return { ...prev, segments: finalSegments };
      });
    }
  };

  const generateVideo = async (index: number) => {
    const currentStory = storyRef.current;
    if (!currentStory) return;
    const segment = currentStory.segments[index];

    const validationError = validateApiKey('video');
    if (validationError) {
      setStory(prev => {
        if (!prev) return prev;
        const updatedSegments = [...prev.segments];
        updatedSegments[index] = { ...updatedSegments[index], error: validationError };
        return { ...prev, segments: updatedSegments };
      });
      return;
    }

    setStory(prev => {
      if (!prev) return prev;
      const updatedSegments = [...prev.segments];
      updatedSegments[index] = { ...updatedSegments[index], isGeneratingVideo: true, error: undefined };
      return { ...prev, segments: updatedSegments };
    });

    if (videoProvider === "Veo") {
      setStory(prev => {
        if (!prev) return prev;
        const finalSegments = [...prev.segments];
        finalSegments[index] = { 
          ...finalSegments[index], 
          isGeneratingVideo: false, 
          error: "Generación de video por IA desactivada para evitar costos. Usa 'Cinematic Pan' en Configuración."
        };
        return { ...prev, segments: finalSegments };
      });
      return;
    }

    if (videoProvider === "CinematicPan") {
      try {
        if (!segment.imageUrl) throw new Error("No hay imagen para animar");
        const duration = visualStyle === "Dios" ? 7000 : 5000;
        const videoBlob = await createCinematicVideoBlob(segment.imageUrl, duration, videoAspectRatio);
        const videoUrl = URL.createObjectURL(videoBlob);

        setStory(prev => {
          if (!prev) return prev;
          const finalSegments = [...prev.segments];
          finalSegments[index] = { 
            ...finalSegments[index], 
            videoUrl, 
            videoBlob,
            isGeneratingVideo: false, 
            error: undefined 
          };
          return { ...prev, segments: finalSegments };
        });
      } catch (err: any) {
        console.error("Cinematic Pan generation error:", err);
        setStory(prev => {
          if (!prev) return prev;
          const finalSegments = [...prev.segments];
          finalSegments[index] = { 
            ...finalSegments[index], 
            isGeneratingVideo: false, 
            error: "Error al generar el efecto Cinematic Pan."
          };
          return { ...prev, segments: finalSegments };
        });
      }
      return;
    }

    try {
      const apiKey = getApiKey();
      if (!apiKey) throw new Error("Por favor, configura tu API Key de Gemini para generar video.");
      const ai = new GoogleGenAI({ apiKey });
      
      // Prepare image as starting frame if it exists
      let imageParam = undefined;
      if (segment.imageUrl && segment.imageUrl.includes(',')) {
        try {
          const base64Data = segment.imageUrl.split(',')[1];
          imageParam = {
            imageBytes: base64Data,
            mimeType: 'image/png'
          };
        } catch (e) {
          console.error("Error processing image for video generation", e);
        }
      } else if (segment.imageUrl) {
        console.warn("Image URL exists but is not a valid data URL for video generation.");
      }

      // Use higher quality model for Pro/Extreme if possible
      const modelName = (modelQuality === "Pro") 
        ? 'veo-3.1-generate-preview' 
        : 'veo-3.1-fast-generate-preview';

      let operation = await withRetry(() => ai.models.generateVideos({
        model: modelName,
        prompt: `ANIMATE THIS SCENE STRICTLY FOLLOWING EVERY DETAIL OF THIS DESCRIPTION: ${segment.imagePrompt}. High quality, cinematic motion, 4k, professional lighting. Silent video, no background music, no voices.`,
        image: imageParam,
        config: { numberOfVideos: 1, resolution: '720p', aspectRatio: videoAspectRatio }
      }));

      let pollCount = 0;
      const maxPolls = 120; // Increased to 10 minutes (120 * 5s) for higher quality model
      
      while (!operation.done && pollCount < maxPolls) {
        await new Promise(resolve => setTimeout(resolve, 5000));
        operation = await withRetry(() => ai.operations.getVideosOperation({ operation: operation }));
        pollCount++;
      }

      if (pollCount >= maxPolls && !operation.done) {
        throw new Error("La generación de video tardó demasiado tiempo (tiempo de espera agotado).");
      }

      const downloadLink = operation.response?.generatedVideos?.[0]?.video?.uri;
      if (downloadLink) {
        addCost(COSTS.VIDEO_GEN, 'videos');
        const videoResponse = await fetch(downloadLink, {
          method: 'GET',
          headers: { 'x-goog-api-key': getApiKey()! },
        });
        
        if (!videoResponse.ok) {
          throw new Error(`Error al descargar el video: ${videoResponse.statusText}`);
        }

        const contentType = videoResponse.headers.get('Content-Type');
        const blob = await videoResponse.blob();
        
        // Validation: Check if it's actually a video and not a small JSON error
        if (contentType && !contentType.includes('video')) {
          const text = await blob.text();
          console.error("Video download returned non-video content:", contentType, text);
          throw new Error("El servidor devolvió un archivo inválido (posible error de API). Intenta de nuevo.");
        }

        if (blob.size < 1000) {
          const text = await blob.text();
          console.error("Downloaded blob is too small, likely an error:", text);
          throw new Error("El video descargado es demasiado pequeño y parece estar corrupto.");
        }

        const videoUrl = URL.createObjectURL(blob);

        setStory(prev => {
          if (!prev) return prev;
          const finalSegments = [...prev.segments];
          finalSegments[index] = { 
            ...finalSegments[index], 
            videoUrl, 
            videoBlob: blob,
            isGeneratingVideo: false, 
            error: undefined 
          };
          return { ...prev, segments: finalSegments };
        });
      } else {
        const opError = (operation as any).error;
        throw new Error(opError?.message || "El servidor no devolvió un enlace de video.");
      }
    } catch (err: any) {
      console.error("Video generation error:", err);
      const errStr = JSON.stringify(err).toLowerCase();
      const isSafetyError = errStr.includes("safety") || err.message?.toLowerCase().includes("safety");
      const isQuotaError = errStr.includes("quota") || errStr.includes("429") || errStr.includes("exhausted");
      const isInternalError = errStr.includes("internal error") || errStr.includes("500") || err.message?.toLowerCase().includes("internal");
      
      let errorMessage = err.message || "Error en la generación de video.";
      if (isSafetyError) {
        errorMessage = "Video bloqueado (Filtro de seguridad). El contenido de la imagen o el prompt es sensible.";
      } else if (isQuotaError) {
        errorMessage = "Límite de Google alcanzado (Quota Exceeded). Has generado muchos videos en poco tiempo. Por favor, espera unos minutos o una hora antes de intentar de nuevo.";
      } else if (isInternalError) {
        errorMessage = "Error interno de Google (Servidor saturado). Reintenta en unos segundos.";
      }
      
      setStory(prev => {
        if (!prev) return prev;
        const finalSegments = [...prev.segments];
        finalSegments[index] = { 
          ...finalSegments[index], 
          isGeneratingVideo: false, 
          error: errorMessage
        };
        return { ...prev, segments: finalSegments };
      });
    }
  };

  const generateAllImages = async (storyToUse?: Story, force = false) => {
    if (storyToUse) {
      setStory(storyToUse);
    }
    
    const activeStory = storyRef.current;
    if (!activeStory || isBulkGeneratingImages) return;
    
    setIsBulkGeneratingImages(true);
    setBulkMode(prev => prev || "images");
    setBulkProgress({ current: 0, total: activeStory.segments.length });
    try {
      const segmentsCount = activeStory.segments.length;
      for (let i = 0; i < segmentsCount; i++) {
        setBulkProgress(prev => prev ? { ...prev, current: i + 1 } : null);
        
        const currentStory = storyRef.current;
        if (!currentStory) break;
        
        const latestSegment = currentStory.segments[i];
        if (latestSegment && (!latestSegment.imageUrl || force)) {
          try {
            // Update UI to show we are actively working on this specific segment
            setStory(prev => {
              if (!prev) return prev;
              const nextSegments = [...prev.segments];
              nextSegments[i] = { ...nextSegments[i], error: "Generando ahora..." };
              return { ...prev, segments: nextSegments };
            });

            await generateImage(i);
            
            // If there are more images to generate, show a countdown for the next one
            if (i < segmentsCount - 1) {
              let countdown = 20; // Increased to 20s to be safer with Google AI Free Tier
              while (countdown > 0) {
                const currentIdx = i;
                setStory(prev => {
                  if (!prev) return prev;
                  const nextSegments = [...prev.segments];
                  if (nextSegments[currentIdx + 1] && !nextSegments[currentIdx + 1].imageUrl) {
                    nextSegments[currentIdx + 1] = { 
                      ...nextSegments[currentIdx + 1], 
                      error: `Esperando turno de Google (${countdown}s)...` 
                    };
                  }
                  return { ...prev, segments: nextSegments };
                });
                await new Promise(resolve => setTimeout(resolve, 1000));
                countdown--;
              }
            }

            // Clear any temporary messages
            setStory(prev => {
              if (!prev) return prev;
              const nextSegments = [...prev.segments];
              if (nextSegments[i + 1] && nextSegments[i + 1].error?.includes("Esperando")) {
                nextSegments[i + 1] = { ...nextSegments[i + 1], error: null };
              }
              return { ...prev, segments: nextSegments };
            });
          } catch (err) {
            console.error(`Error generating image ${i}:`, err);
          }
        }
      }
    } finally {
      setIsBulkGeneratingImages(false);
      if (bulkMode !== "all") setBulkMode(null);
      setBulkProgress(null);
    }
  };

  const generateAllVideos = async (force = false) => {
    const activeStory = storyRef.current;
    if (!activeStory || isBulkGeneratingVideos) return;
    
    setIsBulkGeneratingVideos(true);
    setBulkMode(prev => prev || "videos");
    setBulkProgress({ current: 0, total: activeStory.segments.length });
    try {
      const segmentsCount = activeStory.segments.length;
      
      for (let i = 0; i < segmentsCount; i++) {
        setBulkProgress(prev => prev ? { ...prev, current: i + 1 } : null);
        const currentStory = storyRef.current;
        if (!currentStory) break;

        const latestSegment = currentStory.segments[i];
        // Only animate if it has an image and doesn't have a video yet (unless force or has error)
        if (latestSegment && latestSegment.imageUrl && (!latestSegment.videoUrl || force || latestSegment.error)) {
            try {
            await generateVideo(i);
            // Wait longer for video generation as it's more resource intensive
            // Increased to 10s to better respect API quotas
            await new Promise(resolve => setTimeout(resolve, 10000));
          } catch (err) {
            console.error(`Error animating segment ${i}:`, err);
          }
        }
      }
    } catch (err) {
      console.error("Bulk video generation error:", err);
    } finally {
      setIsBulkGeneratingVideos(false);
      if (bulkMode !== "all") setBulkMode(null);
      setBulkProgress(null);
    }
  };

  const processEverything = async () => {
    if (!storyRef.current || isBulkProcessing) return;
    
    setIsBulkProcessing(true);
    setBulkMode("all");
    try {
      // Step 1: Generate all missing images
      await generateAllImages();
      
      // Step 2: Animate all images
      await generateAllVideos();

      // Step 3: Generate narration audio
      await generateAudio();
    } finally {
      setIsBulkProcessing(false);
      setBulkMode(null);
    }
  };

  const retryAllFailed = async () => {
    const currentStory = storyRef.current;
    if (!currentStory || isBulkProcessing) return;

    const failedIndices = currentStory.segments.reduce((acc, seg, idx) => {
      if (seg.error) acc.push(idx);
      return acc;
    }, [] as number[]);

    if (failedIndices.length === 0) return;

    setIsBulkProcessing(true);
    setBulkMode("all");
    setBulkProgress({ current: 0, total: failedIndices.length });

    try {
      for (let i = 0; i < failedIndices.length; i++) {
        const idx = failedIndices[i];
        setBulkProgress({ current: i + 1, total: failedIndices.length });
        
        const seg = storyRef.current?.segments[idx];
        if (!seg) continue;

        if (seg.error?.toLowerCase().includes("imagen") || !seg.imageUrl) {
          await generateImage(idx);
          // 12 second delay between retries to clear rate limits (Free Tier)
          await new Promise(resolve => setTimeout(resolve, 12000));
        } else {
          await generateVideo(idx);
          // 8 second delay for videos
          await new Promise(resolve => setTimeout(resolve, 8000));
        }
      }
    } finally {
      setIsBulkProcessing(false);
      setBulkMode(null);
      setBulkProgress(null);
    }
  };

  const deleteStoryFromDb = async (storyId: string) => {
    if (!user) return;
    
    const confirmed = await requestConfirm({
      title: "Borrar Historia",
      message: "¿Estás seguro de borrar esta historia de la base de datos? Esta acción no se puede deshacer.",
      confirmText: "Borrar",
      type: "danger"
    });

    if (!confirmed) return;

    try {
      await deleteDoc(doc(db, 'users', user.uid, 'stories', storyId));
    } catch (err) {
      handleFirestoreError(err, OperationType.DELETE, `users/${user.uid}/stories/${storyId}`);
    }
  };

  const loadStoryFromDb = (s: Story & { id: string }) => {
    // Reset generation flags in case it was saved while generating
    const sanitizedStory = {
      ...s,
      segments: s.segments.map(seg => ({
        ...seg,
        isGeneratingImage: false,
        isGeneratingVideo: false
      }))
    };
    setStory(sanitizedStory);
    setCurrentStoryId(s.id);
    setShowMyStories(false);
    setActiveSegmentIndex(0);
  };

  const clearAllAssets = async () => {
    if (!story) return;
    
    const confirmed = await requestConfirm({
      title: "Limpiar Assets",
      message: "¿Estás seguro de que quieres borrar todas las imágenes y videos generados? Esta acción no se puede deshacer.",
      confirmText: "Limpiar",
      type: "danger"
    });

    if (!confirmed) return;

    setStory(prev => {
      if (!prev) return prev;
      return {
        ...prev,
        audioUrl: undefined,
        audioBlob: undefined,
        segments: prev.segments.map(s => ({
          ...s,
          imageUrl: undefined,
          videoUrl: undefined,
          videoBlob: undefined,
          audioBlob: undefined,
          audioUrl: undefined,
          error: undefined,
          isGeneratingImage: false,
          isGeneratingVideo: false
        }))
      };
    });
    setVisualAnchorImages([]);
    setVisualAnchor(null);
  };

  const formatRetryDelay = (delayStr: string) => {
    if (!delayStr) return null;
    const seconds = parseInt(delayStr.replace('s', ''));
    if (isNaN(seconds)) return null;
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    if (hours > 0) {
      return `${hours} hora${hours !== 1 ? 's' : ''} y ${minutes} minuto${minutes !== 1 ? 's' : ''}`;
    }
    return `${minutes} minuto${minutes !== 1 ? 's' : ''}`;
  };

  useEffect(() => {
    if (elevenLabsApiKey && elevenLabsApiKey.length > 10 && elevenLabsVoices.length === 0) {
      fetchElevenLabsVoices();
    }
  }, [elevenLabsApiKey]);

  const fetchElevenLabsVoices = async () => {
    setIsFetchingVoices(true);
    try {
      const response = await axios.get('/api/tts/elevenlabs/voices', {
        headers: { 'xi-api-key': elevenLabsApiKey.trim() }
      });
      const voices = response.data.voices.map((v: any) => ({ id: v.voice_id, name: v.name }));
      setElevenLabsVoices(voices);
      localStorage.setItem('app_elevenlabs_voices', JSON.stringify(voices));
      if (voices.length > 0 && !selectedElevenLabsVoice) {
        setSelectedElevenLabsVoice(voices[0].id);
      }
    } catch (err: any) {
      console.error("Error fetching ElevenLabs voices:", err);
      let msg = "Error al cargar voces";
      const detail = err.response?.data?.detail;
      
      if (detail?.status === "missing_permissions") {
        msg = "La clave no tiene permisos suficientes. Crea una clave SIN restricciones o activa 'voices_read' y 'tts_write' en ElevenLabs.";
      } else if (detail?.message) {
        msg = detail.message;
      } else if (err.response?.data?.error) {
        msg = err.response.data.error;
      } else if (err.response?.status === 401) {
        msg = "API Key Inválida";
      } else if (err.response?.status === 429) {
        msg = "Límite de ElevenLabs superado";
      }
      
      setError(`ElevenLabs: ${msg}`);
    } finally {
      setIsFetchingVoices(false);
    }
  };

  const generateElevenLabsAudio = async (text: string, voiceId: string): Promise<AudioBuffer> => {
    try {
      const response = await axios.post(
        `/api/tts/elevenlabs/${voiceId}`,
        {
          text,
          model_id: "eleven_multilingual_v2",
          voice_settings: {
            stability: 0.5,
            similarity_boost: 0.75
          }
        },
        {
          headers: {
            'xi-api-key': elevenLabsApiKey.trim(),
            'Content-Type': 'application/json'
          },
          responseType: 'arraybuffer'
        }
      );

      const AudioContextClass = (window as any).AudioContext || (window as any).webkitAudioContext;
      const tempCtx = new AudioContextClass();
      const audioBuffer = await tempCtx.decodeAudioData(response.data);
      await tempCtx.close();
      return audioBuffer;
    } catch (err: any) {
      console.error("Error generating ElevenLabs audio:", err);
      let msg = "Error al generar audio";
      
      // Handle arraybuffer error response
      if (err.response?.data instanceof ArrayBuffer) {
        try {
          const decoder = new TextDecoder('utf-8');
          const errorData = JSON.parse(decoder.decode(err.response.data));
          const detail = errorData.detail;
          if (detail?.status === "missing_permissions") {
            msg = "La clave no tiene permisos para generar audio (activa 'tts_write' en ElevenLabs)";
          } else if (detail?.message) {
            msg = detail.message;
          } else if (errorData.error) {
            msg = errorData.error;
          }
        } catch (e) {
          // Fallback to status code
        }
      } else {
        const detail = err.response?.data?.detail;
        if (detail?.status === "missing_permissions") {
          msg = "La clave no tiene permisos para generar audio (activa 'tts_write' en ElevenLabs)";
        } else if (detail?.message) {
          msg = detail.message;
        } else if (err.response?.data?.error) {
          msg = err.response.data.error;
        }
      }

      if (err.response?.status === 401) msg = "API Key Inválida";
      else if (err.response?.status === 429) msg = "Límite de ElevenLabs superado";
      
      throw new Error(`ElevenLabs: ${msg}`);
    }
  };

  const generateAudio = async () => {
    const currentStory = storyRef.current;
    if (!currentStory) {
      console.warn("No story found in storyRef when calling generateAudio");
      return;
    }
    
    console.log("Starting generateAudio for story:", currentStory.title);
    setIsGeneratingAudio(true);
    setError(null);
    
    try {
      const validationError = validateApiKey('audio');
      if (validationError) {
        console.warn("API Key validation failed for audio:", validationError);
        throw new Error(validationError);
      }
      
      if (currentStory.segments.length === 0) {
        console.warn("No segments found to generate audio.");
        setIsGeneratingAudio(false);
        return;
      }

      console.log(`Generating audio for ${currentStory.segments.length} segments.`);

      const apiKey = getApiKey();
      const ai = apiKey ? new GoogleGenAI({ apiKey }) : null;
      
      setAudioProgress({ current: 0, total: currentStory.segments.length });
      const audioBuffers: AudioBuffer[] = [];
      const updatedSegments: StorySegment[] = [];
      const AudioContextClass = (window as any).AudioContext || (window as any).webkitAudioContext;
      const tempCtx = new AudioContextClass();

      let currentOffset = 0;

      for (let i = 0; i < currentStory.segments.length; i++) {
        console.log(`Processing segment ${i + 1}/${currentStory.segments.length}...`);
        setAudioProgress({ current: i, total: currentStory.segments.length });
        const seg = currentStory.segments[i];
        let audioBuffer: AudioBuffer | null = null;
        const textToRead = seg.text.trim();
        
        if (!textToRead) {
           updatedSegments.push({ ...seg, duration: 0, audioStart: currentOffset, audioEnd: currentOffset });
           continue;
        }

        let targetVoice = voiceProvider === 'elevenlabs' ? selectedElevenLabsVoice : selectedVoice;
        
        if (seg.characterVoice) {
           if (voiceProvider === 'elevenlabs') {
              const matchedVoice = elevenLabsVoices.find(v => v.name.toLowerCase() === seg.characterVoice?.trim().toLowerCase());
              if (matchedVoice) targetVoice = matchedVoice.id;
           } else {
              targetVoice = seg.characterVoice.trim();
           }
        }
        
        if (voiceProvider === 'elevenlabs') {
          if (!targetVoice) throw new Error("Voz de ElevenLabs no seleccionada");
          audioBuffer = await generateElevenLabsAudio(textToRead, targetVoice);
        } else {
          if (!ai) throw new Error("API Key de Gemini no configurada para narración.");
          const response = await withRetry(() => ai.models.generateContent({
            model: "gemini-2.5-flash-preview-tts",
            contents: [{ parts: [{ text: textToRead }] }],
            config: {
              responseModalities: [Modality.AUDIO],
              seed: 42,
              speechConfig: {
                voiceConfig: {
                  prebuiltVoiceConfig: { voiceName: targetVoice },
                },
              },
            },
          }));

          const parts = response.candidates?.[0]?.content?.parts || [];
          let base64 = "";
          for (const part of parts) {
            if (part.inlineData?.data) {
              base64 = part.inlineData.data;
              break;
            }
          }

          if (base64) {
            try {
              const { blob } = pcmToWav(base64, 24000);
              const arrayBuffer = await blob.arrayBuffer();
              audioBuffer = await tempCtx.decodeAudioData(arrayBuffer);
            } catch (decodeErr: any) {
              throw new Error(`Error al decodificar el audio del fragmento ${i + 1}`);
            }
          } else {
            throw new Error(`No se recibió audio para el fragmento ${i + 1}`);
          }
        }

        if (audioBuffer) {
           audioBuffers.push(audioBuffer);
           const segDuration = audioBuffer.duration;
           updatedSegments.push({
             ...seg,
             duration: segDuration,
             audioStart: currentOffset,
             audioEnd: currentOffset + segDuration
           });
           currentOffset += segDuration;
        }
      }

      if (audioBuffers.length === 0) throw new Error("No se pudo generar el audio (texto vacío)");

      console.log("Concatenating audio buffers...");
      const totalLength = audioBuffers.reduce((sum, b) => sum + b.length, 0);
      const finalBuffer = tempCtx.createBuffer(
        audioBuffers[0].numberOfChannels,
        totalLength,
        audioBuffers[0].sampleRate
      );

      let offset = 0;
      for (const b of audioBuffers) {
        for (let channel = 0; channel < b.numberOfChannels; channel++) {
          finalBuffer.getChannelData(channel).set(b.getChannelData(channel), offset);
        }
        offset += b.length;
      }

      console.log("Converting final buffer to WAV blob...");
      const wavBlob = await audioBufferToWavBlob(finalBuffer);
      const url = URL.createObjectURL(wavBlob);

      setStory(prev => {
        if (!prev) return null;
        return { 
          ...prev, 
          audioUrl: url, 
          audioBlob: wavBlob,
          isCustomAudio: false,
          segments: updatedSegments 
        };
      });
      
      await tempCtx.close();
      addCost(COSTS.AUDIO_GEN * currentStory.segments.length, 'audios');
      setAudioProgress({ current: currentStory.segments.length, total: currentStory.segments.length });
      console.log("Audio generation completed successfully.");
    } catch (err: any) {
      console.error("Audio generation error:", err);
      setError(`Error al generar la narración: ${err.message || "Error desconocido"}.`);
    } finally {
      setIsGeneratingAudio(false);
      setAudioProgress(null);
    }
  };

  const handleCustomAudioUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    
    const url = URL.createObjectURL(file);
    const audioCtx = new ((window as any).AudioContext || (window as any).webkitAudioContext)();
    const arrayBuffer = await file.arrayBuffer();
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
    const duration = audioBuffer.duration;
    await audioCtx.close();

    setStory(prev => {
      if (!prev) return null;
      
      // Redistribute segments based on the new audio duration (Precise heuristic)
      const segmentMetrics = prev.segments.map(s => {
        const charCount = s.text.length;
        const wordCount = s.text.split(/\s+/).filter(w => w.length > 0).length;
        const sentenceCount = (s.text.match(/[.!?]+/g) || []).length;
        return (charCount * 0.5) + (wordCount * 2.8) + (sentenceCount * 1.5); 
      });

      const totalWeight = segmentMetrics.reduce((sum, w) => sum + w, 0);
      let currentOffset = 0;
      const updatedSegments = prev.segments.map((s, idx) => {
        const proportion = segmentMetrics[idx] / (totalWeight || 1);
        const segDuration = proportion * duration;
        const start = currentOffset;
        currentOffset += segDuration;
        
        return {
          ...s,
          duration: segDuration,
          audioStart: start,
          audioEnd: currentOffset
        };
      });

      return {
        ...prev,
        audioUrl: url,
        audioBlob: file,
        isCustomAudio: true,
        segments: updatedSegments
      };
    });
  };

  const previewVoice = async () => {
    setIsPreviewingVoice(true);
    try {
      const apiKey = getApiKey();
      if (!apiKey) {
        throw new Error("No se detectó ninguna clave API válida. Configura 'GEMINI_API_KEY' en 'Secrets' o selecciona una clave de pago.");
      }
      const ai = new GoogleGenAI({ apiKey });
      const response = await withRetry(() => ai.models.generateContent({
        model: "gemini-2.5-flash-preview-tts",
        contents: [{ parts: [{ text: `Hola, soy la voz ${selectedVoice}. Estoy lista para narrar tu historia.` }] }],
        config: {
          responseModalities: [Modality.AUDIO],
          seed: 42,
          speechConfig: {
            voiceConfig: {
              prebuiltVoiceConfig: { voiceName: selectedVoice },
            },
          },
        },
      }));

      const base64Audio = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
      if (base64Audio) {
        const { url } = pcmToWav(base64Audio, 24000);
        const audio = new Audio(url);
        audio.play();
      }
    } catch (err: any) {
      console.error("Voice preview error:", err);
      const errorStr = JSON.stringify(err).toLowerCase();

      let retryMsg = "";
      try {
        const errorObj = typeof err === 'string' ? JSON.parse(err) : (err.message ? JSON.parse(err.message) : err);
        const details = errorObj?.error?.details || errorObj?.details;
        if (Array.isArray(details)) {
          const retryInfo = details.find((d: any) => d['@type'] === 'type.googleapis.com/google.rpc.RetryInfo');
          if (retryInfo?.retryDelay) {
            const formatted = formatRetryDelay(retryInfo.retryDelay);
            if (formatted) retryMsg = ` Faltan aproximadamente ${formatted} para el reinicio.`;
          }
        }
      } catch (e) {}

      if (errorStr.includes("quota") || errorStr.includes("429") || errorStr.includes("limit")) {
        setError(`Has alcanzado el límite diario de generación de audio (100 peticiones) de la versión gratuita de Google.${retryMsg}`);
      } else {
        setError("Error al previsualizar la voz.");
      }
    } finally {
      setIsPreviewingVoice(false);
    }
  };

  const exportSingleVideo = async () => {
    console.log("Iniciando exportación de video...");
    if (!story) {
      console.error("No hay historia cargada para exportar.");
      return;
    }
    
    setIsExportingVideo(true);
    setExportProgress(0);
    setError(null);
    setExportSceneInfo({ current: 0, total: story?.segments.length || 0 });

    try {
      if (!story || !story.segments || story.segments.length === 0) {
        throw new Error("No hay escenas para exportar");
      }
      console.log("Iniciando exportación de video para:", story.title, "Escenas:", story.segments.length);

      // Basic API support checks
      if (typeof window === 'undefined') throw new Error("Entorno no válido");
      
      const canvas = document.createElement('canvas');
      if (!canvas.captureStream) {
        throw new Error("Tu navegador no soporta la captura de video (canvas.captureStream no disponible). Prueba con Chrome o Edge.");
      }

      const ctx = canvas.getContext('2d', { alpha: false });
      if (!ctx) throw new Error("No se pudo crear el contexto del canvas");

      if (typeof MediaRecorder === 'undefined') {
        throw new Error("Tu navegador no soporta la grabación de video (MediaRecorder no disponible).");
      }

      // Set resolution based on aspect ratio
      let width = 1280;
      let height = 720;
      if (videoAspectRatio === "9:16") {
        width = 720;
        height = 1280;
      } else if (videoAspectRatio === "1:1") {
        width = 1024;
        height = 1024;
      } else if (videoAspectRatio === "4:3") {
        width = 1024;
        height = 768;
      }

      canvas.width = width;
      canvas.height = height;

      // Check supported mime types - Prioritizing WebM for internal recording as it's more stable in browsers
      // We can still name the output .mp4 for the user if it's a compatible container
      const mimeTypes = [
        'video/webm;codecs=vp9,opus',
        'video/webm;codecs=vp8,opus',
        'video/webm',
        'video/mp4;codecs="avc1.42E01E, mp4a.40.2"',
        'video/mp4'
      ];
      let selectedMimeType = '';
      for (const type of mimeTypes) {
        if (MediaRecorder.isTypeSupported(type)) {
          selectedMimeType = type;
          console.log("MimeType seleccionado:", type);
          break;
        }
      }
      
      if (!selectedMimeType) {
        selectedMimeType = 'video/webm';
      }

      console.log("MimeType seleccionado:", selectedMimeType);
      console.log("IMPORTANTE: Mantén esta pestaña enfocada durante la exportación para asegurar la mejor calidad y duración.");

      const stream = canvas.captureStream(30);
      const AudioContextClass = (window as any).AudioContext || (window as any).webkitAudioContext;
      if (!AudioContextClass) throw new Error("Tu navegador no soporta el procesamiento de audio necesario para la exportación.");
      
      const audioCtx = new AudioContextClass();
      if (audioCtx.state === 'suspended') await audioCtx.resume();
      const audioDest = audioCtx.createMediaStreamDestination();

      // Keep AudioContext alive with a silent oscillator
      const silentOsc = audioCtx.createOscillator();
      const silentGain = audioCtx.createGain();
      silentGain.gain.value = 0;
      silentOsc.connect(silentGain);
      silentGain.connect(audioDest);
      silentOsc.start();

      const audioTracks = audioDest.stream.getAudioTracks();
      if (audioTracks.length > 0) {
        stream.addTrack(audioTracks[0]);
      }

      let recorder: MediaRecorder;
      try {
        recorder = new MediaRecorder(stream, {
          mimeType: selectedMimeType,
          videoBitsPerSecond: 5000000,
          audioBitsPerSecond: 128000
        });
      } catch (e) {
        console.warn("Error al crear MediaRecorder con mimeType específico, intentando por defecto:", e);
        recorder = new MediaRecorder(stream);
      }

      const chunks: Blob[] = [];
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.push(e.data);
      };

      const exportPromise = new Promise<Blob>((resolve, reject) => {
        recorder.onstop = () => {
          stream.getTracks().forEach(track => track.stop());
          resolve(new Blob(chunks, { type: recorder.mimeType || selectedMimeType }));
        };
        recorder.onerror = (e) => {
          console.error("MediaRecorder error:", e);
          reject(new Error("Error en el grabador de video"));
        };
      });

      stopExportRef.current = false;

      // Global narration audio preparation
      let fullAudioBuffer: AudioBuffer | null = null;
      if (story.audioUrl) {
        try {
          const response = await fetch(story.audioUrl);
          const arrayBuffer = await response.arrayBuffer();
          fullAudioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
          console.log("Full audio narration loaded for export slicing");
        } catch (e) {
          console.warn("Error loading global audio for export:", e);
        }
      }

      // Background music preparation
      let bgMusicSource: AudioBufferSourceNode | null = null;
      if (story.backgroundMusicUrl) {
        try {
          const bgMusicData = await fetch(story.backgroundMusicUrl).then(r => r.arrayBuffer());
          const bgMusicBuffer = await audioCtx.decodeAudioData(bgMusicData);
          bgMusicSource = audioCtx.createBufferSource();
          bgMusicSource.buffer = bgMusicBuffer;
          bgMusicSource.loop = true;
          const bgMusicGain = audioCtx.createGain();
          bgMusicGain.gain.value = 0.15;
          bgMusicSource.connect(bgMusicGain);
          bgMusicGain.connect(audioDest);
          console.log("Background music prepared for export");
        } catch (e) {
          console.warn("No se pudo cargar la música de fondo para la exportación:", e);
        }
      }

      // Apply global audio offset if needed
      let globalOffsetDelay = globalAudioOffset;

      let isRecorderStarted = false;

      for (let i = 0; i < story.segments.length; i++) {
        if (stopExportRef.current) break;
        const seg = story.segments[i];
        setExportSceneInfo({ current: i + 1, total: story.segments.length });
        setExportProgress(Math.round(((i + 1) / story.segments.length) * 100));
        console.log(`Procesando escena ${i + 1}/${story.segments.length}`);

        // Pre-cargar la siguiente escena si existe
        let nextVideo: HTMLVideoElement | null = null;
        if (i < story.segments.length - 1) {
          const nextSeg = story.segments[i + 1];
          if (nextSeg.videoUrl && nextSeg.videoUrl.length > 0 && nextSeg.videoUrl !== nextSeg.imageUrl) {
            nextVideo = document.createElement('video');
            nextVideo.src = nextSeg.videoUrl;
            nextVideo.muted = true;
            nextVideo.crossOrigin = "anonymous";
            nextVideo.preload = "auto";
            nextVideo.load();
          }
        }

        // NO dibujar texto de carga en el canvas que se está grabando
        // Solo limpiar con negro si es necesario
        if (i === 0 && !isRecorderStarted) {
          ctx.fillStyle = 'black';
          ctx.fillRect(0, 0, width, height);
        }

        let duration = seg.duration || (visualStyle === "Dios" ? 7 : 5); 
        let audioSource: AudioBufferSourceNode | null = null;

        // Load or slice audio for this segment
        if (fullAudioBuffer && seg.audioStart !== undefined && seg.audioEnd !== undefined) {
          const startSample = Math.floor(seg.audioStart * fullAudioBuffer.sampleRate);
          const endSample = Math.floor(seg.audioEnd * fullAudioBuffer.sampleRate);
          const sliceLength = endSample - startSample;
          
          if (sliceLength > 0) {
            const sliceBuffer = audioCtx.createBuffer(
              fullAudioBuffer.numberOfChannels,
              sliceLength,
              fullAudioBuffer.sampleRate
            );
            for (let ch = 0; ch < fullAudioBuffer.numberOfChannels; ch++) {
              sliceBuffer.getChannelData(ch).set(
                fullAudioBuffer.getChannelData(ch).subarray(startSample, endSample)
              );
            }
            duration = sliceBuffer.duration;
            audioSource = audioCtx.createBufferSource();
            audioSource.buffer = sliceBuffer;
            audioSource.connect(audioDest);
          }
        } else if (seg.audioUrl && !story.audioUrl) {
          try {
            const response = await fetch(seg.audioUrl);
            const arrayBuffer = await response.arrayBuffer();
            const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
            duration = audioBuffer.duration;
            
            audioSource = audioCtx.createBufferSource();
            audioSource.buffer = audioBuffer;
            audioSource.connect(audioDest);
          } catch (e) {
            console.warn(`Error al cargar audio de escena ${i + 1}:`, e);
          }
        }

        // PAUSAR grabación mientras se carga la escena para evitar desincronización
        // No suspendemos el AudioContext para evitar ruidos y problemas de clock
        if (isRecorderStarted && recorder.state === 'recording') {
          recorder.pause();
          console.log(`Grabación pausada para cargar escena ${i + 1}`);
        }

        const isCinematicPan = !!(seg.videoUrl && seg.imageUrl && seg.videoUrl === seg.imageUrl);
        const isRealVideo = !!(seg.videoUrl && seg.videoUrl.length > 0 && !isCinematicPan);

        if (isRealVideo) {
          console.log(`Cargando video para escena ${i + 1}: ${seg.videoUrl!.substring(0, 50)}...`);
          // Load video
          const video = document.createElement('video');
          video.src = seg.videoUrl;
          video.muted = true;
          video.crossOrigin = "anonymous";
          video.playsInline = true;
          video.preload = "auto";
          video.load();
          
          await new Promise((resolve) => {
            const timeout = setTimeout(() => {
              console.warn(`Timeout cargando video de escena ${i + 1}`);
              resolve(null);
            }, 15000); // Aumentado a 15s

            if (video.readyState >= 3) { // HAVE_FUTURE_DATA
              clearTimeout(timeout);
              resolve(null);
              return;
            }

            video.onloadeddata = () => {
              clearTimeout(timeout);
              resolve(null);
            };
            video.oncanplay = () => {
              clearTimeout(timeout);
              resolve(null);
            };
            video.onerror = (e) => {
              clearTimeout(timeout);
              console.error(`Error cargando video de escena ${i + 1}:`, e);
              resolve(null);
            };
          });

          // Si hay audio (global o por segmento), la duración del audio es la maestra para mantener la sincronización.
          // Solo usamos la duración del video si no hay audio disponible.
          if (!story.audioUrl && !seg.audioUrl) {
            const videoDuration = isFinite(video.duration) ? video.duration : 0;
            duration = Math.max(videoDuration, duration);
          }
          
          if (!isFinite(duration) || duration <= 0 || duration > 3600) duration = visualStyle === "Dios" ? 7 : 5;
          duration = Math.max(0.1, duration);

          console.log(`Escena ${i + 1}: Duración calculada = ${duration}s`);

          try {
            video.currentTime = 0;
            await video.play();
          } catch (e) {
            console.warn(`No se pudo reproducir video de escena ${i + 1}:`, e);
          }
          
          // REANUDAR grabación
          if (isRecorderStarted && recorder.state === 'paused') {
            recorder.resume();
            console.log(`Grabación reanudada para escena ${i + 1}`);
          }

          // Draw first frame before starting recorder to avoid black frame
          try {
            ctx.drawImage(video, 0, 0, width, height);
            if (subtitleStyle !== 'none') drawSubtitles(ctx, seg.text, width, height);
          } catch (e) {}

          if (!isRecorderStarted) {
            recorder.start(500);
            if (bgMusicSource) bgMusicSource.start();
            isRecorderStarted = true;
            console.log("Recorder and bg music started with first video frame");
          }

          if (audioSource) {
            const startTime = Math.max(0, audioCtx.currentTime + (i === 0 ? globalOffsetDelay : 0));
            audioSource.start(startTime);
          }

          console.log(`Escena ${i + 1} (Video): Iniciando renderizado en tiempo real...`);
          const sceneStartTime = performance.now();
          const sceneDurationMs = duration * 1000;

          while (true) {
            const now = performance.now();
            const elapsed = now - sceneStartTime;
            if (elapsed >= sceneDurationMs || stopExportRef.current) break;
            
            if (recorder.state !== 'recording') {
              throw new Error("El grabador de video se detuvo inesperadamente durante la escena de video.");
            }
            
            if (video.paused && !video.ended && video.readyState >= 2) {
              try { await video.play(); } catch(e) {}
            }

            try {
              ctx.drawImage(video, 0, 0, width, height);
              if (subtitleStyle !== 'none') drawSubtitles(ctx, seg.text, width, height);
            } catch (e) {
              // Ignorar errores de frame individual
            }
            
            // Sincronizar con el refresco de pantalla o fallback
            await new Promise(r => {
              const raf = requestAnimationFrame(() => {
                clearTimeout(tm);
                r(null);
              });
              const tm = setTimeout(() => {
                cancelAnimationFrame(raf);
                r(null);
              }, 16); // ~60fps target para suavidad
            });
          }
          video.pause();
          video.src = "";
          video.load();
        } else if (seg.imageUrl || isCinematicPan) {
          const mediaUrl = isCinematicPan ? seg.videoUrl! : seg.imageUrl!;
          console.log(`Cargando imagen para escena ${i + 1}: ${mediaUrl.substring(0, 50)}...`);
          // Load image
          const img = new Image();
          img.src = mediaUrl;
          img.crossOrigin = "anonymous";
          await new Promise((resolve) => {
            const timeout = setTimeout(resolve, 8000);
            if (img.complete) {
              clearTimeout(timeout);
              resolve(null);
              return;
            }
            img.onload = () => {
              clearTimeout(timeout);
              resolve(null);
            };
            img.onerror = () => {
              clearTimeout(timeout);
              console.error(`Error cargando imagen de escena ${i + 1}`);
              resolve(null);
            };
          });

          if (!isFinite(duration) || duration <= 0 || duration > 3600) duration = visualStyle === "Dios" ? 7 : 5;
          duration = Math.max(0.1, duration);
          console.log(`Escena ${i + 1} (Imagen): Duración = ${duration}s`);
          
          // REANUDAR grabación
          if (isRecorderStarted && recorder.state === 'paused') {
            recorder.resume();
            console.log(`Grabación reanudada para escena ${i + 1}`);
          }

          // Draw first frame before starting recorder to avoid black frame
          try {
            const progress = 0;
            const scale = 1 + progress * 0.08;
            const x = (width - width * scale) / 2;
            const y = (height - height * scale) / 2;
            
            ctx.save();
            ctx.translate(x, y);
            ctx.scale(scale, scale);
            ctx.drawImage(img, 0, 0, width, height);
            ctx.restore();

            if (subtitleStyle !== 'none') drawSubtitles(ctx, seg.text, width, height);
          } catch (e) {}

          if (!isRecorderStarted) {
            recorder.start(500);
            if (bgMusicSource) bgMusicSource.start();
            isRecorderStarted = true;
            console.log("Recorder and bg music started with first image frame");
          }

          if (audioSource) {
            const startTime = Math.max(0, audioCtx.currentTime + (i === 0 ? globalOffsetDelay : 0));
            audioSource.start(startTime);
          }

          console.log(`Escena ${i + 1} (Imagen): Iniciando renderizado en tiempo real...`);
          const sceneStartTime = performance.now();
          const sceneDurationMs = duration * 1000;

          while (true) {
            const now = performance.now();
            const elapsed = now - sceneStartTime;
            if (elapsed >= sceneDurationMs || stopExportRef.current) break;

            if (recorder.state !== 'recording') {
              throw new Error("El grabador de video se detuvo inesperadamente durante la escena de imagen.");
            }

            try {
              const progress = elapsed / sceneDurationMs;
              const scale = 1 + progress * 0.08;
              const x = (width - width * scale) / 2;
              const y = (height - height * scale) / 2;
              
              ctx.save();
              ctx.translate(x, y);
              ctx.scale(scale, scale);
              ctx.drawImage(img, 0, 0, width, height);
              ctx.restore();

              if (subtitleStyle !== 'none') drawSubtitles(ctx, seg.text, width, height);
            } catch (e) {
              // Ignorar
            }
            
            await new Promise(r => {
              const raf = requestAnimationFrame(() => {
                clearTimeout(tm);
                r(null);
              });
              const tm = setTimeout(() => {
                cancelAnimationFrame(raf);
                r(null);
              }, 16);
            });
          }
        } else {
          // Escena sin media (solo audio y texto)
          console.log(`Escena ${i + 1} sin media. Duración = ${duration}s`);
          if (!isFinite(duration) || duration <= 0) duration = 5;
          duration = Math.max(0.1, duration);

          // REANUDAR grabación
          if (isRecorderStarted && recorder.state === 'paused') {
            recorder.resume();
            console.log(`Grabación reanudada para escena ${i + 1}`);
          }
          
          if (audioSource) {
            const startTime = Math.max(0, audioCtx.currentTime + (i === 0 ? globalOffsetDelay : 0));
            audioSource.start(startTime);
          }
          
          const sceneStartTime = performance.now();
          const sceneDurationMs = duration * 1000;

          while (true) {
            const now = performance.now();
            const elapsed = now - sceneStartTime;
            if (elapsed >= sceneDurationMs || stopExportRef.current) break;
            
            if (recorder.state !== 'recording') {
              throw new Error("El grabador de video se detuvo inesperadamente durante la escena sin media.");
            }
            
            ctx.fillStyle = 'black';
            ctx.fillRect(0, 0, width, height);
            if (subtitleStyle !== 'none') drawSubtitles(ctx, seg.text, width, height);
            
            await new Promise(r => {
              const raf = requestAnimationFrame(() => {
                clearTimeout(tm);
                r(null);
              });
              const tm = setTimeout(() => {
                cancelAnimationFrame(raf);
                r(null);
              }, 16);
            });
          }
        }

        if (audioSource) {
          try { audioSource.stop(); } catch(e) {}
        }
      }

      if (bgMusicSource) {
        try { bgMusicSource.stop(); } catch(e) {}
      }
      
      if (silentOsc) {
        try { silentOsc.stop(); } catch(e) {}
      }

      if (stopExportRef.current) {
        if (recorder.state === 'recording') recorder.stop();
        try { audioCtx.close(); } catch(e) {}
        setIsExportingVideo(false);
        setExportProgress(0);
        return;
      }
      
      // Finalizar grabación con un delay mínimo para asegurar que se capturen los últimos frames sin añadir tiempo muerto
      await new Promise(r => setTimeout(r, 100));
      
      if (recorder.state === 'recording' || recorder.state === 'paused') {
        recorder.stop();
      }
      
      try { audioCtx.close(); } catch(e) {}
      setExportProgress(100);

      const finalBlob = await exportPromise;
      const url = URL.createObjectURL(finalBlob);
      const link = document.createElement('a');
      link.href = url;
      
      // Determine extension based on actual mimeType
      const actualMimeType = recorder.mimeType || selectedMimeType;
      const isWebM = actualMimeType.includes('webm');
      const extension = isWebM ? 'webm' : 'mp4';
      
      link.download = `${story.title.replace(/\s+/g, '_')}_final.${extension}`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);

      // Show compatibility message if it's WebM
      if (isWebM) {
        setTimeout(() => {
          alert("¡Video exportado con éxito!\n\nNota: El video se guardó como .webm porque tu navegador no permitió crear un .mp4 directamente. Si Windows Media Player no lo abre, te recomendamos usar el reproductor gratuito VLC o simplemente arrastrar el archivo a una pestaña de Chrome/Edge.");
        }, 1000);
      }

      // Success confetti
      import('canvas-confetti').then(confetti => {
        confetti.default({
          particleCount: 150,
          spread: 70,
          origin: { y: 0.6 },
          colors: ['#10b981', '#ffffff', '#34d399']
        });
      });

    } catch (err: any) {
      console.error("Error crítico en exportación:", err);
      setError("Error al exportar el video: " + (err.message || "Error desconocido"));
      // Show error in an alert if it's a critical failure during init
      alert("Error al iniciar la exportación: " + (err.message || "Error desconocido"));
    } finally {
      setIsExportingVideo(false);
      setExportProgress(0);
    }
  };


  // Helper to draw subtitles
  const drawSubtitles = (ctx: CanvasRenderingContext2D, text: string, width: number, height: number) => {
    ctx.save();
    const padding = 40;
    const fontSize = width * 0.04;
    ctx.font = `bold ${fontSize}px Inter, sans-serif`;
    if (subtitleStyle === 'cinematic') ctx.font = `italic ${fontSize}px Georgia, serif`;
    if (subtitleStyle === 'bold') ctx.font = `italic 900 ${fontSize * 1.2}px Inter, sans-serif`;

    const words = text.split(' ');
    const lines = [];
    let currentLine = '';
    for (const word of words) {
      const testLine = currentLine + word + ' ';
      if (ctx.measureText(testLine).width > width - padding * 4) {
        lines.push(currentLine);
        currentLine = word + ' ';
      } else {
        currentLine = testLine;
      }
    }
    lines.push(currentLine);

    const lineHeight = fontSize * 1.2;
    const totalHeight = lines.length * lineHeight;
    let y = height - padding - totalHeight;
    if (subtitlePosition === 'top') y = padding + fontSize;
    if (subtitlePosition === 'middle') y = (height - totalHeight) / 2;

    lines.forEach((line, idx) => {
      const textWidth = ctx.measureText(line).width;
      const x = (width - textWidth) / 2;
      const lineY = y + idx * lineHeight;

      if (subtitleStyle === 'classic') {
        ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
        ctx.fillRect(x - 10, lineY - fontSize, textWidth + 20, lineHeight + 10);
        ctx.fillStyle = 'white';
        ctx.fillText(line, x, lineY);
      } else if (subtitleStyle === 'minimal') {
        ctx.shadowColor = 'rgba(0, 0, 0, 0.8)';
        ctx.shadowBlur = 4;
        ctx.shadowOffsetX = 2;
        ctx.shadowOffsetY = 2;
        ctx.fillStyle = 'white';
        ctx.fillText(line, x, lineY);
      } else if (subtitleStyle === 'bold') {
        ctx.fillStyle = '#facc15'; // yellow-400
        ctx.fillRect(x - 10, lineY - fontSize, textWidth + 20, lineHeight + 10);
        ctx.fillStyle = 'black';
        ctx.fillText(line.toUpperCase(), x, lineY);
      } else if (subtitleStyle === 'cinematic') {
        ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
        ctx.fillText(line, x, lineY);
      }
    });
    ctx.restore();
  };


  const downloadPromptsCSV = () => {
    if (!story) return;
    
    const csvContent = [
        "ID,Prompt Visual",
        ...story.segments.map((seg, i) => 
            `"${i + 1}","${seg.imagePrompt.replace(/"/g, '""')}"`
        )
    ].join('\n');
    
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${story.title.replace(/\s+/g, '_')}_prompts_visuales.csv`;
    link.click();
  };

  const downloadVideoDescriptionsCSV = () => {
    if (!story) return;
    
    const csvContent = [
        "ID,Descripción de Video",
        ...story.segments.map((seg, i) => 
            `"${i + 1}","${seg.videoDescription.replace(/"/g, '""')}"`
        )
    ].join('\n');
    
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${story.title.replace(/\s+/g, '_')}_movimientos_visuales.csv`;
    link.click();
  };

  const downloadAll = async () => {
    if (!story) return;
    const zip = new JSZip();
    const folder = zip.folder(story.title.replace(/\s+/g, '_'));
    
    if (folder) {
      // Add narration text
      folder.file("narration.txt", story.narration);
      
      // Create metadata content
      const metadataContent = [
        `TÍTULO: ${story.title}`,
        `TIPO: ${story.type === 'quote' ? 'Frase' : 'Historia'}`,
        `FECHA: ${new Date(story.createdAt || Date.now()).toLocaleString()}`,
        `HASHTAGS: ${story.hashtags.map(h => h.startsWith('#') ? h : `#${h}`).join(' ')}`,
        `\nNARRACIÓN:\n${story.narration}`
      ].join('\n');
      
      folder.file("metadata_y_hashtags.txt", metadataContent);

      // Add images and videos
      for (let i = 0; i < story.segments.length; i++) {
        const seg = story.segments[i];
        if (seg.imageUrl) {
          if (seg.imageUrl.startsWith('data:')) {
            const imgData = seg.imageUrl.split(',')[1];
            folder.file(`segment_${i + 1}_image.png`, imgData, { base64: true });
          } else {
            try {
              const imgBlob = await fetch(seg.imageUrl).then(r => r.blob());
              folder.file(`segment_${i + 1}_image.png`, imgBlob);
            } catch (e) {
              console.warn(`Error al descargar imagen de escena ${i + 1}:`, e);
            }
          }
        }
        if (seg.videoUrl) {
          const videoBlob = seg.videoBlob || await fetch(seg.videoUrl).then(r => r.blob());
          folder.file(`segment_${i + 1}_video.mp4`, videoBlob);
        }
        if (seg.audioUrl) {
          const audioBlob = seg.audioBlob || await fetch(seg.audioUrl).then(r => r.blob());
          folder.file(`segment_${i + 1}_audio.wav`, audioBlob);
        }
      }

      // Add audio if exists
      if (story.audioUrl) {
        const audioBlob = await fetch(story.audioUrl).then(r => r.blob());
        folder.file("00_NARRACION_COMPLETA.wav", audioBlob);
      }

      const content = await zip.generateAsync({ type: "blob" });
      const url = URL.createObjectURL(content);
      const link = document.createElement('a');
      link.href = url;
      link.download = `${story.title.replace(/\s+/g, '_')}_package.zip`;
      link.click();
    }
  };

  if (isAccessDenied) {
    return (
      <div className="min-h-screen bg-[#0a0a0a] flex items-center justify-center p-6">
        <motion.div 
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="max-w-md w-full bg-zinc-900 border border-red-500/30 p-10 rounded-3xl text-center shadow-2xl shadow-red-500/10"
        >
          <div className="w-20 h-20 bg-red-500/20 rounded-2xl flex items-center justify-center mx-auto mb-8">
            <ShieldAlert className="w-10 h-10 text-red-500" />
          </div>
          <h1 className="text-3xl font-bold text-white mb-4 tracking-tight">Acceso Denegado</h1>
          <p className="text-zinc-400 mb-8 leading-relaxed">
            Esta aplicación es privada. Solo el propietario tiene permiso para acceder a las herramientas de generación.
          </p>
          <button 
            onClick={() => setIsAccessDenied(false)}
            className="w-full py-4 bg-zinc-800 hover:bg-zinc-700 text-white font-bold rounded-xl transition-all"
          >
            Volver al Inicio
          </button>
        </motion.div>
      </div>
    );
  }

  if (!isAuthReady) {
    return (
      <div className="min-h-screen bg-[#0a0a0a] text-white flex items-center justify-center">
        <Loader2 className="w-8 h-8 animate-spin text-emerald-500" />
      </div>
    );
  }

  if (!user) {
    return (
      <div className="min-h-screen bg-[#0a0a0a] text-white flex flex-col items-center justify-center p-6 text-center">
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="max-w-md w-full space-y-8">
          <div className="w-20 h-20 bg-emerald-500/20 rounded-3xl flex items-center justify-center mx-auto border border-emerald-500/30">
            <User className="w-10 h-10 text-emerald-400" />
          </div>
          <div className="space-y-4">
            <h1 className="text-4xl font-bold tracking-tight">Mi Web Original</h1>
            <p className="text-zinc-400 leading-relaxed">
              Inicia sesión para acceder a tus historias privadas y videos personalizados.
            </p>
          </div>
          <button 
            onClick={loginWithGoogle}
            className="w-full py-4 bg-white text-black font-bold rounded-xl hover:bg-zinc-200 transition-all transform active:scale-[0.98] flex items-center justify-center gap-3"
          >
            <img src="https://www.gstatic.com/firebasejs/ui/2.0.0/images/auth/google.svg" className="w-6 h-6" alt="Google" />
            Continuar con Google
          </button>
        </motion.div>
      </div>
    );
  }

  if (!isAuthorized) {
    return (
      <div className="min-h-screen bg-[#0a0a0a] text-white flex flex-col items-center justify-center p-6 text-center">
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="max-w-md w-full space-y-8">
          <div className="w-20 h-20 bg-zinc-900 rounded-3xl flex items-center justify-center mx-auto border border-zinc-800">
            <Lock className="w-10 h-10 text-zinc-400" />
          </div>
          <div className="space-y-4">
            <h1 className="text-4xl font-bold tracking-tight">Mi Web Original</h1>
            <p className="text-zinc-400 leading-relaxed">
              Introduce la clave de acceso para continuar con tu proyecto personalizado.
            </p>
          </div>
          <form onSubmit={handlePasswordSubmit} className="space-y-4">
            <input
              type="password"
              value={passwordInput}
              onChange={(e) => setPasswordInput(e.target.value)}
              placeholder="Clave de acceso"
              className={cn(
                "w-full bg-black border rounded-xl px-4 py-4 text-center text-xl font-mono tracking-widest focus:ring-2 focus:ring-emerald-500/50 outline-none transition-all",
                passwordError ? "border-red-500" : "border-zinc-800"
              )}
            />
            {passwordError && (
              <p className="text-red-500 text-sm font-bold">Clave incorrecta. Inténtalo de nuevo.</p>
            )}
            <button 
              type="submit"
              className="w-full py-4 bg-white text-black font-bold rounded-xl hover:bg-zinc-200 transition-all transform active:scale-[0.98]"
            >
              Entrar
            </button>
          </form>
        </motion.div>
      </div>
    );
  }

  if (!apiKeySelected) {
    if (showApiKeyInput) {
      return (
        <div className="min-h-screen bg-[#0a0a0a] text-white flex flex-col items-center justify-center p-6">
          <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} className="max-w-md w-full bg-zinc-900 border border-zinc-800 p-8 rounded-3xl space-y-6 shadow-2xl">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-emerald-500/20 rounded-xl flex items-center justify-center border border-emerald-500/30">
                <Key className="w-5 h-5 text-emerald-400" />
              </div>
              <h2 className="text-xl font-bold">Configurar API Key</h2>
            </div>
            <p className="text-zinc-400 text-sm">
              Ingresa tu clave API de Gemini aquí. Se guardará localmente en tu navegador.
            </p>
            <form onSubmit={handleManualApiKeySave} className="space-y-4">
              <input
                type="password"
                value={manualApiKey}
                onChange={(e) => setManualApiKey(e.target.value)}
                placeholder="AIzaSy..."
                className="w-full bg-black border border-zinc-800 rounded-xl p-4 text-sm focus:ring-2 focus:ring-emerald-500/50 outline-none transition-all font-mono"
              />
              <div className="space-y-4">
              <div className="flex items-center justify-between p-4 bg-black border border-zinc-800 rounded-xl">
                <div className="flex items-center gap-3">
                  <TypeIcon className="w-4 h-4 text-emerald-400" />
                  <div className="text-left">
                    <div className="text-xs font-bold text-white">Grabar frases en imágenes</div>
                    <div className="text-[9px] text-zinc-500">Inserta el texto automáticamente en cada escena</div>
                  </div>
                </div>
                <button 
                  type="button"
                  onClick={() => {
                    const newVal = !burnTextIntoImages;
                    setBurnTextIntoImages(newVal);
                    localStorage.setItem('app_burn_text', String(newVal));
                  }}
                  className={cn(
                    "w-10 h-5 rounded-full transition-all relative shrink-0",
                    burnTextIntoImages ? "bg-emerald-500" : "bg-zinc-800"
                  )}
                >
                  <div className={cn(
                    "absolute top-1 w-3 h-3 bg-white rounded-full transition-all",
                    burnTextIntoImages ? "right-1" : "left-1"
                  )} />
                </button>
              </div>
              </div>
              <div className="flex gap-3">
                <button 
                  type="button"
                  onClick={() => setShowApiKeyInput(false)}
                  className="flex-1 py-3 bg-zinc-800 hover:bg-zinc-700 text-white font-bold rounded-xl transition-all"
                >
                  Cancelar
                </button>
                <button 
                  type="submit"
                  className="flex-[2] py-3 bg-emerald-500 hover:bg-emerald-600 text-black font-bold rounded-xl transition-all shadow-lg shadow-emerald-500/20"
                >
                  Guardar Clave
                </button>
              </div>
            </form>
          </motion.div>
        </div>
      );
    }

    return (
      <div className="min-h-screen bg-[#0a0a0a] text-white flex flex-col items-center justify-center p-6 text-center">
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="max-w-md space-y-8">
          <div className="w-20 h-20 bg-emerald-500/20 rounded-3xl flex items-center justify-center mx-auto border border-emerald-500/30">
            <Sparkles className="w-10 h-10 text-emerald-400" />
          </div>
          <div className="space-y-4">
            <h1 className="text-4xl font-bold tracking-tight">Visualizador de Historias IA</h1>
            <p className="text-zinc-400 leading-relaxed">
              Para usar la generación de imágenes y videos de alta calidad de forma gratuita, obtén tu propia API Key en Google AI Studio.
            </p>
          </div>
          <div className="space-y-3">
            <a 
              href="https://aistudio.google.com/app/apikey" 
              target="_blank" 
              rel="noopener noreferrer"
              className="w-full py-4 bg-emerald-500 hover:bg-emerald-600 text-black font-bold rounded-xl transition-all transform hover:scale-[1.02] active:scale-[0.98] shadow-lg shadow-emerald-500/20 flex items-center justify-center gap-2"
            >
              <ExternalLink className="w-5 h-5" />
              Obtener API Key Gratis (AI Studio)
            </a>
            <button 
              onClick={() => setShowApiKeyInput(true)} 
              className="w-full py-3 bg-zinc-900/50 hover:bg-zinc-800 text-zinc-300 text-sm font-medium rounded-xl border border-zinc-800 transition-all flex items-center justify-center gap-2"
            >
              <Key className="w-4 h-4" />
              Ingresar Clave Manualmente
            </button>
            <button 
              onClick={() => setApiKeySelected(true)} 
              className="w-full py-3 bg-zinc-900 hover:bg-zinc-800 text-zinc-400 text-sm font-medium rounded-xl transition-all"
            >
              Omitir por ahora (Funciones limitadas)
            </button>
            {!getApiKey() && (
              <p className="text-[10px] text-red-500/50 mt-1">
                Nota: No se detectó clave gratuita. Configúrala en 'Secrets' o ingrésala manualmente.
              </p>
            )}
          </div>
          <p className="text-xs text-zinc-500">
            Requiere un proyecto de Google Cloud con facturación habilitada. 
          </p>
          <div className="pt-4 text-[10px] text-zinc-700 uppercase tracking-widest flex flex-col gap-1">
            <div>Estado: {detectedKey ? `Clave Detectada (${getApiKeySource()})` : "Esperando Clave..."}</div>
            {!detectedKey && (
              <div className="text-[8px] text-zinc-800 font-mono">
                D: {String(!!getVal(window, 'process.env.API_KEY'))}|{String(!!getVal(window, 'API_KEY'))}|{String(!!getVal(window, 'process.env.GEMINI_API_KEY'))}|{String(!!getVal(window, 'GEMINI_API_KEY'))}
              </div>
            )}
          </div>
        </motion.div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-zinc-100 font-sans selection:bg-emerald-500/30">
      {/* Settings Loading Overlay */}
      {user && !isSettingsLoaded && (
        <div className="fixed inset-0 z-[200] bg-[#0a0a0a] flex flex-col items-center justify-center gap-6">
          <div className="relative">
            <div className="w-24 h-24 border-4 border-emerald-500/20 rounded-full animate-pulse" />
            <div className="absolute inset-0 flex items-center justify-center">
              <Loader2 className="w-10 h-10 text-emerald-500 animate-spin" />
            </div>
          </div>
          <div className="text-center">
            <h2 className="text-xl font-bold text-white mb-2 tracking-tight">Sincronizando con la Nube</h2>
            <p className="text-sm text-zinc-400 max-w-xs mx-auto leading-relaxed">
              Estamos recuperando tus preferencias y proyectos guardados...
            </p>
          </div>
        </div>
      )}

      {/* Quota Warning Overlay */}
      {isQuotaExceeded && (
        <div className="fixed bottom-6 right-6 z-[100] max-w-md animate-in slide-in-from-right-8 duration-500">
          <div className="bg-zinc-900 border border-amber-500/50 rounded-2xl p-4 shadow-2xl shadow-amber-500/10 backdrop-blur-xl">
            <div className="flex items-start gap-4">
              <div className="p-2 bg-amber-500/20 rounded-lg">
                <AlertTriangle className="w-5 h-5 text-amber-500" />
              </div>
              <div className="flex-1">
                <h3 className="text-sm font-bold text-amber-500 mb-1">Límite de API Alcanzado</h3>
                <p className="text-xs text-zinc-400 leading-relaxed mb-3">
                  {quotaErrorMessage}
                </p>
                <div className="flex gap-2">
                  <button 
                    onClick={() => setIsQuotaExceeded(false)}
                    className="px-3 py-1.5 bg-zinc-800 hover:bg-zinc-700 text-zinc-300 text-[10px] font-bold rounded-lg transition-colors"
                  >
                    Entendido
                  </button>
                  <button 
                    onClick={() => {
                      setIsQuotaExceeded(false);
                      // Trigger settings modal if available, or just scroll to top
                      window.scrollTo({ top: 0, behavior: 'smooth' });
                    }}
                    className="px-3 py-1.5 bg-amber-500/20 hover:bg-amber-500/30 text-amber-500 text-[10px] font-bold rounded-lg transition-colors border border-amber-500/30"
                  >
                    Cambiar API Key
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Bulk Progress Bar */}
      <AnimatePresence>
        {(isBulkGeneratingImages || isBulkGeneratingVideos) && bulkProgress && (
          <motion.div 
            initial={{ opacity: 0, y: -100 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -100 }}
            className="fixed top-20 left-0 right-0 z-[100] px-6"
          >
            <div className="max-w-xl mx-auto bg-zinc-900/90 backdrop-blur-xl border border-emerald-500/30 rounded-2xl p-4 shadow-2xl shadow-emerald-500/10">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-emerald-500/20 rounded-lg">
                    <Loader2 className="w-4 h-4 text-emerald-500 animate-spin" />
                  </div>
                  <div>
                    <h4 className="text-xs font-bold text-white uppercase tracking-widest">
                      {isBulkGeneratingVideos ? "Generando Videos Pro..." : "Generando Imágenes..."}
                    </h4>
                    <p className="text-[10px] text-zinc-500">Procesando escena {bulkProgress.current} de {bulkProgress.total}</p>
                  </div>
                </div>
                <span className="text-xs font-bold text-emerald-500">{Math.round((bulkProgress.current / bulkProgress.total) * 100)}%</span>
              </div>
              <div className="w-full h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                <motion.div 
                  className="h-full bg-emerald-500"
                  initial={{ width: 0 }}
                  animate={{ width: `${(bulkProgress.current / bulkProgress.total) * 100}%` }}
                />
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Header */}
      <div className="bg-emerald-500/10 border-b border-emerald-500/20 py-1.5 px-4 flex items-center justify-center gap-2">
        <ShieldCheck className="w-3 h-3 text-emerald-500" />
        <span className="text-[10px] font-bold uppercase tracking-[0.2em] text-emerald-500">Modo Gratuito Activo - Solo Modelos Flash</span>
      </div>
      <header className="border-b border-zinc-800/50 bg-black/50 backdrop-blur-xl sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 h-20 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="w-10 h-10 bg-emerald-500 rounded-xl flex items-center justify-center shadow-lg shadow-emerald-500/20">
              <Sparkles className="w-6 h-6 text-black" />
            </div>
            <div>
              <h1 className="font-bold text-lg tracking-tight">Mi Web Original</h1>
              <div className="flex items-center gap-2">
                <span className="text-[10px] font-bold text-emerald-500 uppercase tracking-widest">v2.1 Premium</span>
                <span className="w-1 h-1 bg-zinc-700 rounded-full" />
                <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest">Base de Datos Lista</span>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <button 
              onClick={() => setShowMyStories(!showMyStories)}
              className="px-4 py-2 bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 rounded-xl text-xs font-bold transition-all flex items-center gap-2"
            >
              <BookOpen className="w-4 h-4" />
              Mis Historias
            </button>
            
            <div className="h-8 w-px bg-zinc-800 mx-1" />

            <div className="flex items-center gap-3 bg-zinc-900/50 p-1.5 pr-4 rounded-xl border border-zinc-800">
              <img src={user.photoURL || ""} className="w-8 h-8 rounded-lg" alt="Profile" />
              <div className="hidden sm:block">
                <div className="text-[10px] font-bold text-white truncate max-w-[100px]">{user.displayName}</div>
                <button onClick={logout} className="text-[9px] font-bold text-zinc-500 hover:text-red-400 transition-colors uppercase tracking-wider">Cerrar Sesión</button>
              </div>
            </div>
          </div>
        </div>
        {validateApiKey('story') && (
          <div className="bg-red-500/10 border-b border-red-500/20 py-2 px-6 flex items-center justify-center gap-2 text-[10px] font-bold text-red-400 uppercase tracking-widest">
            <AlertCircle className="w-3 h-3" />
            {validateApiKey('story')}
          </div>
        )}
        <div className="max-w-7xl mx-auto px-6 h-12 flex items-center justify-end gap-2 border-t border-zinc-800/30">
          <div className="flex items-center gap-2">
            <button 
              className="px-4 py-1.5 rounded-full text-xs font-bold bg-white text-black transition-all"
            >
              Crear
            </button>
            <div className="w-px h-4 bg-zinc-800 mx-2" />
            <button 
              onClick={resetProject}
              className="px-4 py-1.5 rounded-full text-xs font-bold text-red-400 hover:bg-red-500/10 transition-all border border-red-500/20"
            >
              Nuevo Proyecto
            </button>
            <div className="w-px h-4 bg-zinc-800 mx-2" />
            <button onClick={() => setApiKeySelected(false)} className="p-2 hover:bg-zinc-800 rounded-full transition-colors">
              <Settings className="w-5 h-5 text-zinc-400" />
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-4 relative">
        {/* My Stories Sidebar/Overlay */}
        <AnimatePresence>
          {showMyStories && (
            <motion.div 
              initial={{ opacity: 0, x: 300 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 300 }}
              className="fixed inset-y-0 right-0 w-full sm:w-96 bg-zinc-950 border-l border-zinc-800 z-[60] shadow-2xl p-6 flex flex-col"
            >
              <div className="flex items-center justify-between mb-8">
                <h2 className="text-xl font-bold flex items-center gap-2">
                  <BookOpen className="w-5 h-5 text-emerald-500" />
                  Mis Historias
                </h2>
                <button onClick={() => setShowMyStories(false)} className="p-2 hover:bg-zinc-900 rounded-lg">
                  <X className="w-5 h-5" />
                </button>
              </div>

              <div className="flex-1 overflow-y-auto space-y-4 pr-2 custom-scrollbar">
                {userStories.length === 0 ? (
                  <div className="text-center py-12">
                    <div className="w-12 h-12 bg-zinc-900 rounded-2xl flex items-center justify-center mx-auto mb-4 border border-zinc-800">
                      <Anchor className="w-6 h-6 text-zinc-600" />
                    </div>
                    <p className="text-sm text-zinc-500 italic">No tienes historias guardadas aún.</p>
                  </div>
                ) : (
                  userStories.map((s) => (
                    <div 
                      key={s.id} 
                      className="group p-4 bg-zinc-900/50 border border-zinc-800 rounded-2xl hover:border-emerald-500/50 transition-all cursor-pointer relative"
                      onClick={() => loadStoryFromDb(s)}
                    >
                      <div className="flex items-center gap-2 mb-1">
                        <span className={cn(
                          "text-[8px] font-black uppercase tracking-widest px-1.5 py-0.5 rounded-md",
                          s.type === 'quote' ? "bg-amber-500/20 text-amber-500 border border-amber-500/30" : "bg-emerald-500/20 text-emerald-500 border border-emerald-500/30"
                        )}>
                          {s.type === 'quote' ? 'Frase' : 'Historia'}
                        </span>
                        <h3 className="font-bold text-sm truncate pr-8 flex-1">{s.title}</h3>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-[10px] text-zinc-500">{new Date(s.createdAt || 0).toLocaleDateString()}</span>
                        <span className="text-[10px] font-bold text-emerald-500 uppercase tracking-widest">{s.segments.length} Escenas</span>
                      </div>
                      <button 
                        onClick={(e) => {
                          e.stopPropagation();
                          deleteStoryFromDb(s.id);
                        }}
                        className="absolute top-4 right-4 p-1.5 bg-red-500/10 text-red-500 rounded-lg opacity-0 group-hover:opacity-100 transition-opacity hover:bg-red-500/20"
                      >
                        <X className="w-3 h-3" />
                      </button>
                    </div>
                  ))
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          {/* Left Column: Input & Controls */}
        <div className="lg:col-span-4 space-y-4 lg:sticky lg:top-8 max-h-[calc(100vh-4rem)] overflow-y-auto pr-2 custom-scrollbar">
          <section 
            className="bg-zinc-900/50 border border-zinc-800 rounded-2xl p-4 space-y-4"
          >
            <div className="space-y-4">
              {/* Sidebar Tabs */}
              <div className="flex p-1 bg-black/40 rounded-xl border border-zinc-800/50">
                    <button
                      onClick={() => {
                        setSidebarTab('narrative');
                        localStorage.setItem('app_sidebar_tab', 'narrative');
                      }}
                      className={cn(
                        "flex-1 flex items-center justify-center gap-2 py-2 text-[10px] font-black uppercase tracking-widest rounded-lg transition-all",
                        sidebarTab === 'narrative' 
                          ? "bg-emerald-500 text-black shadow-lg shadow-emerald-500/20" 
                          : "text-zinc-500 hover:text-zinc-300"
                      )}
                    >
                      <BookOpen className="w-3.5 h-3.5" />
                      Historias
                    </button>
                    <button
                      onClick={() => {
                        setSidebarTab('quote');
                        localStorage.setItem('app_sidebar_tab', 'quote');
                      }}
                      className={cn(
                        "flex-1 flex items-center justify-center gap-2 py-2 text-[10px] font-black uppercase tracking-widest rounded-lg transition-all",
                        sidebarTab === 'quote' 
                          ? "bg-amber-500 text-black shadow-lg shadow-amber-500/20" 
                          : "text-zinc-500 hover:text-zinc-300"
                      )}
                    >
                      <Zap className="w-3.5 h-3.5" />
                      Frases
                    </button>
                    <button
                      onClick={() => {
                        setSidebarTab('analyze');
                        localStorage.setItem('app_sidebar_tab', 'analyze');
                      }}
                      className={cn(
                        "flex-1 flex items-center justify-center gap-2 py-2 text-[10px] font-black uppercase tracking-widest rounded-lg transition-all",
                        sidebarTab === 'analyze' 
                          ? "bg-blue-500 text-black shadow-lg shadow-blue-500/20" 
                          : "text-zinc-500 hover:text-zinc-300"
                      )}
                    >
                      <Video className="w-3.5 h-3.5" />
                      Analizar
                    </button>
                  </div>

                  <AnimatePresence mode="wait">
                    {sidebarTab === 'analyze' ? (
                      <motion.div
                        key="analyze-tab"
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -10 }}
                        className="space-y-4 pt-2"
                      >
                        <div className="p-4 bg-blue-500/5 border border-blue-500/20 rounded-2xl space-y-3">
                          <div className="flex items-center gap-2 text-blue-500">
                            <Video className="w-4 h-4" />
                            <h3 className="text-[11px] font-black uppercase tracking-wider">Analizador de Estilo</h3>
                          </div>
                          <p className="text-[10px] text-zinc-400 leading-relaxed">
                            Sube un video para que la IA lo analice y extraiga su esencia visual, narrativa y emocional. Podrás usar este análisis para crear contenido similar.
                          </p>
                          
                          <div className="relative group">
                            <input 
                              type="file" 
                              accept="video/*"
                              onChange={async (e) => {
                                const file = e.target.files?.[0];
                                if (!file) return;
                                
                                setIsAnalyzingVideo(true);
                                setAnalysisResult(null);
                                setError(null);
                                
                                try {
                                  const apiKey = getApiKey()!;
                                  const ai = new GoogleGenAI({ apiKey });
                                  
                                  const reader = new FileReader();
                                  reader.readAsDataURL(file);
                                  reader.onload = async () => {
                                    try {
                                      const base64 = (reader.result as string).split(',')[1];
                                      
                                      const response = await ai.models.generateContent({
                                        model: "gemini-3.1-pro-preview",
                                        contents: [
                                          {
                                            inlineData: {
                                              data: base64,
                                              mimeType: file.type
                                            }
                                          },
                                          {
                                            text: "Analiza este video detalladamente. Extrae: 1. Un resumen de la historia/acción. 2. El estilo visual exacto (colores, iluminación, técnica). 3. El tono narrativo (sensual, épico, melancólico, etc.). 4. Un prompt maestro que pueda usar para recrear este estilo y narrativa. Devuelve el resultado en formato JSON con los campos: 'summary', 'style', 'tone', 'prompt'."
                                          }
                                        ],
                                        config: {
                                          responseMimeType: "application/json",
                                          responseSchema: {
                                            type: Type.OBJECT,
                                            properties: {
                                              summary: { type: Type.STRING },
                                              style: { type: Type.STRING },
                                              tone: { type: Type.STRING },
                                              prompt: { type: Type.STRING }
                                            },
                                            required: ["summary", "style", "tone", "prompt"]
                                          }
                                        }
                                      });
                                      
                                      const result = JSON.parse(response.text);
                                      setAnalysisResult(result);
                                      setIsAnalyzingVideo(false);
                                    } catch (err) {
                                      console.error("Error processing video analysis:", err);
                                      setError("Error al procesar el análisis del video.");
                                      setIsAnalyzingVideo(false);
                                    }
                                  };
                                } catch (err) {
                                  console.error("Error starting video analysis:", err);
                                  setError("Error al iniciar el análisis.");
                                  setIsAnalyzingVideo(false);
                                }
                              }}
                              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                            />
                            <div className="w-full py-8 border-2 border-dashed border-zinc-800 rounded-xl flex flex-col items-center justify-center gap-2 group-hover:border-blue-500/50 transition-all bg-black/40">
                              {isAnalyzingVideo ? (
                                <Loader2 className="w-6 h-6 text-blue-500 animate-spin" />
                              ) : (
                                <Upload className="w-6 h-6 text-zinc-600 group-hover:text-blue-500 transition-colors" />
                              )}
                              <span className="text-[10px] font-bold text-zinc-500 group-hover:text-zinc-300">
                                {isAnalyzingVideo ? "Analizando Video..." : "Subir Video para Analizar"}
                              </span>
                            </div>
                          </div>
                        </div>

                        {analysisResult && (
                          <motion.div 
                            initial={{ opacity: 0, scale: 0.95 }}
                            animate={{ opacity: 1, scale: 1 }}
                            className="p-4 bg-zinc-900 border border-zinc-800 rounded-2xl space-y-4"
                          >
                            <div className="space-y-1">
                              <label className="text-[9px] font-black text-blue-500 uppercase tracking-widest">Estilo Detectado</label>
                              <p className="text-[11px] text-white font-bold">{analysisResult.style}</p>
                            </div>
                            <div className="space-y-1">
                              <label className="text-[9px] font-black text-blue-500 uppercase tracking-widest">Tono Narrativo</label>
                              <p className="text-[11px] text-zinc-300">{analysisResult.tone}</p>
                            </div>
                            <div className="space-y-1">
                              <label className="text-[9px] font-black text-blue-500 uppercase tracking-widest">Prompt Maestro</label>
                              <div className="relative group">
                                <p className="text-[10px] text-zinc-400 italic bg-black/40 p-2 rounded-lg border border-zinc-800 line-clamp-3">
                                  {analysisResult.prompt}
                                </p>
                                <button 
                                  onClick={() => {
                                    setPrompt(analysisResult.prompt);
                                    setSidebarTab('narrative');
                                  }}
                                  className="absolute top-2 right-2 p-1.5 bg-blue-500 text-black rounded-md opacity-0 group-hover:opacity-100 transition-opacity"
                                >
                                  <Copy className="w-3 h-3" />
                                </button>
                              </div>
                            </div>
                            <button 
                              onClick={() => {
                                setPrompt(analysisResult.summary);
                                setVisualStyle("Comic Realista Moderno");
                                setSidebarTab('narrative');
                              }}
                              className="w-full py-2 bg-blue-500 hover:bg-blue-400 text-black font-black text-[10px] uppercase tracking-widest rounded-xl transition-all"
                            >
                              Usar este Análisis
                            </button>
                          </motion.div>
                        )}
                      </motion.div>
                    ) : sidebarTab === 'quote' ? (
                      <motion.div
                        key="quote-tab"
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -10 }}
                        className="space-y-4"
                      >
                        <div className="space-y-2">
                          <label className="text-[10px] font-bold uppercase tracking-wider text-zinc-500 flex items-center gap-2">
                            <Zap className="w-3 h-3 text-amber-400" /> Temas Populares
                          </label>
                          <div className="grid grid-cols-2 gap-1.5">
                            {QUICK_THEMES.map((theme) => (
                              <button
                                key={theme.name}
                                onClick={() => handleQuickTheme(theme)}
                                disabled={isGeneratingStory || isBulkGeneratingImages}
                                className="px-2 py-2 rounded-lg text-[9px] font-bold bg-zinc-900/80 border border-zinc-800 text-zinc-400 hover:border-amber-500/50 hover:text-amber-400 hover:bg-amber-500/5 transition-all disabled:opacity-50 disabled:cursor-not-allowed text-left flex flex-col gap-0.5"
                              >
                                <span className="text-zinc-200">{theme.name}</span>
                                <span className="text-[7px] font-medium text-zinc-600 line-clamp-1">{theme.prompt}</span>
                              </button>
                            ))}
                          </div>
                        </div>

                        <div className="space-y-2">
                          <label className="text-[10px] font-bold uppercase tracking-wider text-zinc-500 flex items-center gap-2">
                            <Sparkles className="w-3 h-3 text-amber-400" /> Tu propia frase
                          </label>
                          <textarea
                            value={prompt}
                            onChange={(e) => {
                              setPrompt(e.target.value);
                              localStorage.setItem('app_prompt', e.target.value);
                            }}
                            placeholder="Escribe el tema de tus frases..."
                            className="w-full bg-black border border-zinc-800 rounded-xl p-3 text-xs focus:ring-2 focus:ring-amber-500/50 focus:border-amber-500 outline-none transition-all min-h-[60px] resize-none"
                          />
                          <div className="space-y-1.5">
                            <label className="text-[10px] font-bold uppercase tracking-wider text-zinc-500">Cantidad</label>
                            <select 
                              value={storyDuration}
                              onChange={(e) => setStoryDuration(e.target.value as StoryDuration)}
                              className="w-full bg-black border border-zinc-800 rounded-lg px-2 py-1.5 text-[10px] font-bold text-white outline-none focus:border-amber-500 transition-all"
                            >
                              <option value="Auto">Auto (Completo)</option>
                              <optgroup label="Imágenes" className="bg-zinc-900">
                                <option value="10imgs">10 Imágenes</option>
                                <option value="20imgs">20 Imágenes</option>
                                <option value="30imgs">30 Imágenes</option>
                                <option value="40imgs">40 Imágenes</option>
                                <option value="50imgs">50 Imágenes</option>
                                <option value="100imgs">100 Imágenes</option>
                              </optgroup>
                            </select>
                          </div>
                          <button
                            onClick={() => generateStory(undefined, undefined, 'quote')}
                            disabled={isGeneratingStory || !prompt.trim()}
                            className="w-full py-3 bg-amber-500 hover:bg-amber-400 disabled:bg-zinc-800 text-black font-black text-xs uppercase tracking-widest rounded-xl transition-all shadow-lg shadow-amber-500/20 flex items-center justify-center gap-2"
                          >
                            {isGeneratingStory ? <Loader2 className="w-4 h-4 animate-spin" /> : <Zap className="w-4 h-4" />}
                            Generar Frases
                          </button>
                        </div>
                      </motion.div>
                    ) : (
                      <motion.div
                        key="narrative-tab"
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -10 }}
                        className="space-y-4"
                      >
                        <div className="space-y-2">
                          <div className="flex items-center justify-between">
                            <label className="text-[10px] font-bold uppercase tracking-wider text-zinc-500 flex items-center gap-2">
                              <BookOpen className="w-3 h-3" /> Concepto de la Historia
                            </label>
                            <button 
                              onClick={() => {
                                const newVal = !useExactText;
                                setUseExactText(newVal);
                                localStorage.setItem('app_use_exact_text', String(newVal));
                              }}
                              className={cn(
                                "text-[9px] font-bold px-1.5 py-0.5 rounded-full border transition-all flex items-center gap-1",
                                useExactText ? "bg-emerald-500/20 border-emerald-500 text-emerald-500" : "bg-zinc-900 border-zinc-800 text-zinc-500"
                              )}
                            >
                              <CheckCircle2 className="w-2 h-2" /> Texto Exacto
                            </button>
                          </div>
                          <textarea
                            value={prompt}
                            onChange={(e) => {
                              setPrompt(e.target.value);
                              localStorage.setItem('app_prompt', e.target.value);
                            }}
                            placeholder={useExactText ? "Pega aquí tu historia..." : "Describe tu historia..."}
                            className="w-full bg-black border border-zinc-800 rounded-xl p-3 text-xs focus:ring-2 focus:ring-emerald-500/50 focus:border-emerald-500 outline-none transition-all min-h-[80px] resize-none"
                          />
                        </div>

                        {visualAnchor && (
                          <div className="p-3 bg-emerald-500/5 border border-emerald-500/20 rounded-xl space-y-2">
                            <div className="flex items-center justify-between">
                              <span className="text-[9px] font-bold text-emerald-500 uppercase tracking-widest flex items-center gap-1">
                                <Zap className="w-2.5 h-2.5" /> Aprendizaje Activo
                              </span>
                              <button 
                                onClick={() => {
                                  setVisualAnchor(null);
                                  setVisualAnchorImages([]);
                                  localStorage.removeItem('app_visual_anchor');
                                  localStorage.removeItem('app_visual_anchor_images');
                                }}
                                className="text-[9px] text-zinc-500 hover:text-zinc-300 underline"
                              >
                                Reiniciar
                              </button>
                            </div>
                            <p className="text-[10px] text-zinc-400 line-clamp-2 italic">"{visualAnchor}"</p>
                          </div>
                        )}

                        <div className="grid grid-cols-2 gap-3">
                          <div className="space-y-1.5">
                            <label className="text-[10px] font-bold uppercase tracking-wider text-zinc-500">Cámara</label>
                            <button
                              onClick={() => setDynamicAngles(!dynamicAngles)}
                              className={cn(
                                "w-full py-1.5 rounded-lg text-[10px] font-bold border transition-all flex items-center justify-center gap-2",
                                dynamicAngles ? "bg-emerald-500 border-emerald-500 text-black" : "bg-black border-zinc-800 text-zinc-400 hover:border-zinc-600"
                              )}
                            >
                              <Video className="w-3 h-3" />
                              {dynamicAngles ? "Dinámica" : "Fija"}
                            </button>
                          </div>
                          <div className="space-y-1.5">
                            <label className="text-[10px] font-bold uppercase tracking-wider text-zinc-500">Cantidad</label>
                            <select 
                              value={storyDuration}
                              onChange={(e) => setStoryDuration(e.target.value as StoryDuration)}
                              className="w-full bg-black border border-zinc-800 rounded-lg px-2 py-1.5 text-[10px] font-bold text-white outline-none focus:border-emerald-500 transition-all"
                            >
                              <option value="Auto">Auto (Completo)</option>
                              <optgroup label="Tiempo" className="bg-zinc-900">
                                <option value="30s">30 Segundos</option>
                                <option value="1m">1 Minuto</option>
                                <option value="2m">2 Minutos</option>
                                <option value="3m">3 Minutos</option>
                              </optgroup>
                              <optgroup label="Imágenes" className="bg-zinc-900">
                                <option value="10imgs">10 Imágenes</option>
                                <option value="20imgs">20 Imágenes</option>
                                <option value="30imgs">30 Imágenes</option>
                                <option value="40imgs">40 Imágenes</option>
                                <option value="50imgs">50 Imágenes</option>
                                <option value="100imgs">100 Imágenes</option>
                              </optgroup>
                            </select>
                          </div>
                        </div>

                          <button
                            onClick={() => generateStory(undefined, undefined, 'narrative')}
                            disabled={isGeneratingStory || !prompt.trim()}
                            className="w-full py-4 bg-emerald-500 hover:bg-emerald-400 disabled:bg-zinc-800 text-black font-black text-xs uppercase tracking-widest rounded-xl transition-all shadow-lg shadow-emerald-500/20 flex items-center justify-center gap-2"
                          >
                            {isGeneratingStory ? <Loader2 className="w-4 h-4 animate-spin" /> : (useExactText ? <Search className="w-4 h-4" /> : <Sparkles className="w-4 h-4" />)}
                            {isGeneratingStory 
                              ? (useExactText ? "Analizando..." : "Redactando...") 
                              : (useExactText ? "Analizar Historia" : "Generar Historia")}
                          </button>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>

                {savedCharacters.length > 0 && (
                  <div className="space-y-2">
                    <label className="text-[10px] font-bold uppercase tracking-wider text-zinc-600">Biblioteca de Personajes (Memoria)</label>
                    <div className="flex flex-wrap gap-2">
                      {savedCharacters.map((char) => (
                        <div 
                          key={char.id}
                          className="group relative"
                        >
                          <button
                            onClick={() => {
                              setCharacterDesc(char.description);
                              setVisualStyle(char.style);
                            }}
                            className={cn(
                              "px-3 py-1.5 rounded-lg text-[10px] font-bold border transition-all flex items-center gap-2",
                              characterDesc === char.description ? "bg-emerald-500/20 border-emerald-500 text-emerald-500" : "bg-zinc-900 border-zinc-800 text-zinc-400 hover:border-zinc-600"
                            )}
                          >
                            <User className="w-3 h-3" />
                            {char.name}
                          </button>
                          <button 
                            onClick={() => deleteCharacter(char.id)}
                            className="absolute -top-1 -right-1 w-4 h-4 bg-red-500 text-white rounded-full flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
                          >
                            <X className="w-2 h-2" />
                          </button>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {visualAnchorImages.length > 0 && (
                  <div className="space-y-2 p-3 bg-emerald-500/5 border border-emerald-500/20 rounded-xl">
                    <div className="flex items-center justify-between">
                      <label className="text-[10px] font-bold uppercase tracking-wider text-emerald-500 flex items-center gap-1">
                        <Anchor className="w-3 h-3" /> Avatares de Consistencia ({visualAnchorImages.length})
                      </label>
                      <button 
                        onClick={() => {
                          setVisualAnchor(null);
                          setVisualAnchorImages([]);
                        }}
                        className="text-[10px] text-zinc-500 hover:text-red-400 transition-colors"
                      >
                        Limpiar Todo
                      </button>
                    </div>
                    <div className="grid grid-cols-3 gap-2">
                      {visualAnchorImages.map((img, idx) => (
                        <div key={idx} className="relative aspect-square rounded-lg overflow-hidden border border-emerald-500/30 group">
                          <img src={img} className="w-full h-full object-cover" alt={`Avatar ${idx + 1}`} referrerPolicy="no-referrer" />
                          <button 
                            onClick={() => deleteAvatar(idx)}
                            className="absolute top-1 right-1 w-5 h-5 bg-red-500 text-white rounded-full flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity shadow-lg"
                          >
                            <X className="w-3 h-3" />
                          </button>
                        </div>
                      ))}
                      <div className="relative aspect-square rounded-lg border border-dashed border-zinc-800 hover:border-emerald-500/50 transition-all bg-black/40 flex items-center justify-center group cursor-pointer">
                        <input 
                          type="file" 
                          accept="image/*"
                          multiple
                          onChange={handleAvatarUpload}
                          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                        />
                        <UserPlus className="w-5 h-5 text-zinc-600 group-hover:text-emerald-500 transition-colors" />
                      </div>
                    </div>
                    <p className="text-[9px] text-zinc-500 italic">
                      Estas imágenes se usan como referencia para mantener la consistencia del personaje.
                    </p>
                  </div>
                )}

                {visualAnchorImages.length === 0 && (
                  <div className="relative group">
                    <input 
                      type="file" 
                      accept="image/*"
                      multiple
                      onChange={handleAvatarUpload}
                      className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                    />
                    <div className="w-full py-3 border border-dashed border-zinc-800 rounded-xl flex items-center justify-center gap-2 group-hover:border-emerald-500/50 transition-all bg-black/40">
                      <UserPlus className="w-4 h-4 text-zinc-600 group-hover:text-emerald-500 transition-colors" />
                      <span className="text-[10px] font-bold text-zinc-500 group-hover:text-zinc-300">
                        Añadir Avatar (Consistencia)
                      </span>
                    </div>
                  </div>
                )}

                <div className="space-y-4">
                  <div className="space-y-2">
                    <label className="text-xs font-bold uppercase tracking-wider text-zinc-500">Proveedor de Imágenes</label>
                    <div className="grid grid-cols-2 gap-2">
                      {(["Google", "Pollinations"] as const).map((p) => (
                        <button
                          key={p}
                          onClick={() => setImageProvider(p)}
                          className={cn(
                            "py-2 rounded-lg text-xs font-bold border transition-all flex flex-col items-center",
                            imageProvider === p ? "bg-emerald-500 border-emerald-500 text-black" : "bg-black border-zinc-800 text-zinc-400 hover:border-zinc-600"
                          )}
                        >
                          <span>{p === "Google" ? "Google AI (Flash)" : "Pollinations"}</span>
                          <span className="text-[8px] opacity-70">{p === "Google" ? "Gratis (Flash 2.5)" : "Gratis (Ilimitado)"}</span>
                        </button>
                      ))}
                    </div>
                  </div>

                  <div className="space-y-2">
                    <label className="text-xs font-bold uppercase tracking-wider text-zinc-500">Calidad del Modelo (Costo)</label>
                    <div className="grid grid-cols-2 gap-2">
                      {(["Pro", "Savings"] as ModelQuality[]).map((q) => (
                        <button
                          key={q}
                          onClick={() => setModelQuality(q)}
                          className={cn(
                            "py-2 rounded-lg text-[10px] font-bold border transition-all flex flex-col items-center",
                            modelQuality === q ? "bg-emerald-500 border-emerald-500 text-black" : "bg-black border-zinc-800 text-zinc-400 hover:border-zinc-600"
                          )}
                        >
                          <span>{q === "Pro" ? "Pro (Key)" : "Ahorro (Key)"}</span>
                          <span className="text-[8px] opacity-70">Google Gemini</span>
                        </button>
                      ))}
                    </div>
                  </div>

                  <div className="space-y-2">
                    <label className="text-xs font-bold uppercase tracking-wider text-zinc-500">Resolución de Imagen</label>
                    <div className="grid grid-cols-3 gap-2">
                      {(["1K", "2K", "4K"] as ImageSize[]).map((size) => (
                        <button
                          key={size}
                          onClick={() => setImageSize(size)}
                          className={cn(
                            "py-2 rounded-lg text-xs font-bold border transition-all",
                            imageSize === size ? "bg-emerald-500 border-emerald-500 text-black" : "bg-black border-zinc-800 text-zinc-400 hover:border-zinc-600"
                          )}
                        >
                          {size}
                        </button>
                      ))}
                    </div>
                  </div>

                  <div className="space-y-4">
                    <div className="space-y-2">
                      <label className="text-xs font-bold uppercase tracking-wider text-zinc-500">Proveedor de Video (Costo)</label>
                      <div className="grid grid-cols-2 gap-2">
                        {(["Veo", "CinematicPan"] as const).map((p) => (
                          <button
                            key={p}
                            onClick={() => setVideoProvider(p)}
                            className={cn(
                              "py-2 rounded-lg text-xs font-bold border transition-all flex flex-col items-center",
                              videoProvider === p ? "bg-emerald-500 border-emerald-500 text-black" : "bg-black border-zinc-800 text-zinc-400 hover:border-zinc-600"
                            )}
                          >
                            <span>{p === "Veo" ? "Google Veo (Desactivado)" : "Cinematic Pan (Local)"}</span>
                            <span className="text-[8px] opacity-70">{p === "Veo" ? "Evitar Costos" : "Gratis (Ilimitado)"}</span>
                          </button>
                        ))}
                      </div>
                      <p className="text-[9px] text-zinc-500 font-medium leading-tight px-1">
                        * Cinematic Pan ahora genera un archivo de video real (.mp4) grabando la animación en tu navegador.
                      </p>
                    </div>

                    <div className="space-y-4">
                      <div className="space-y-2">
                        <label className="text-xs font-bold uppercase tracking-wider text-zinc-500 flex items-center justify-between">
                          <span>Sincronización de Audio (Global)</span>
                          <span className={cn("text-[10px] font-mono", globalAudioOffset === 0 ? "text-zinc-500" : "text-emerald-500")}>
                            {globalAudioOffset > 0 ? `+${globalAudioOffset.toFixed(1)}s` : `${globalAudioOffset.toFixed(1)}s`}
                          </span>
                        </label>
                        <div className="flex items-center gap-3 bg-black border border-zinc-800 rounded-xl px-4 py-3">
                          <button 
                            onClick={() => setGlobalAudioOffset(prev => Math.max(-5, prev - 0.1))}
                            className="w-8 h-8 flex items-center justify-center bg-zinc-900 hover:bg-zinc-800 text-zinc-400 hover:text-white rounded-lg transition-all border border-zinc-800"
                          >
                            -
                          </button>
                          <div className="flex-1 flex flex-col items-center">
                            <input 
                              type="range" 
                              min="-5" 
                              max="5" 
                              step="0.1"
                              value={globalAudioOffset}
                              onChange={(e) => setGlobalAudioOffset(parseFloat(e.target.value))}
                              className="w-full accent-emerald-500 h-1.5 bg-zinc-800 rounded-lg appearance-none cursor-pointer"
                            />
                            <div className="flex justify-between w-full mt-1 text-[8px] text-zinc-600 font-bold uppercase tracking-widest">
                              <span>Tarde (-5s)</span>
                              <span>Sync</span>
                              <span>Pronto (+5s)</span>
                            </div>
                          </div>
                          <button 
                            onClick={() => setGlobalAudioOffset(prev => Math.min(5, prev + 0.1))}
                            className="w-8 h-8 flex items-center justify-center bg-zinc-900 hover:bg-zinc-800 text-zinc-400 hover:text-white rounded-lg transition-all border border-zinc-800"
                          >
                            +
                          </button>
                        </div>
                        <p className="text-[9px] text-zinc-500 italic leading-tight px-1">
                          * Ajusta este valor si notas que la voz empieza antes o después que el video. 
                          Valores negativos retrasan el audio, positivos lo adelantan.
                        </p>
                      </div>

                      <div className="space-y-2">
                        <label className="text-xs font-bold uppercase tracking-wider text-zinc-500">Proveedor de Voz</label>
                        <div className="grid grid-cols-2 gap-2">
                          {(["gemini", "elevenlabs"] as const).map((p) => (
                            <button
                              key={p}
                              onClick={() => setVoiceProvider(p)}
                              className={cn(
                                "py-2 rounded-lg text-xs font-bold border transition-all flex flex-col items-center",
                                voiceProvider === p ? "bg-emerald-500 border-emerald-500 text-black" : "bg-black border-zinc-800 text-zinc-400 hover:border-zinc-600"
                              )}
                            >
                              <span>{p === "gemini" ? "Google Gemini" : "ElevenLabs"}</span>
                              <span className="text-[8px] opacity-70">{p === "gemini" ? "Gratis (Flash 2.5)" : "Premium (Créditos Eleven)"}</span>
                            </button>
                          ))}
                        </div>
                      </div>

                      {voiceProvider === 'elevenlabs' ? (
                        <div className="space-y-3">
                          <div className="space-y-1.5">
                            <label className="text-[10px] font-bold uppercase tracking-wider text-zinc-500 flex items-center justify-between">
                              <span>ElevenLabs API Key</span>
                              <a href="https://elevenlabs.io/app/settings/api-keys" target="_blank" rel="noreferrer" className="text-emerald-500 hover:underline">Obtener Clave</a>
                            </label>
                            <div className="relative">
                              <Key className="absolute left-3 top-1/2 -translate-y-1/2 w-3 h-3 text-zinc-500" />
                              <input 
                                type="password"
                                value={elevenLabsApiKey}
                                onChange={(e) => setElevenLabsApiKey(e.target.value)}
                                placeholder="sk_..."
                                className="w-full bg-black border border-zinc-800 rounded-lg pl-9 pr-3 py-2 text-[10px] font-bold text-white outline-none focus:border-emerald-500 transition-all"
                              />
                            </div>
                          </div>
                          <div className="space-y-1.5">
                            <label className="text-[10px] font-bold uppercase tracking-wider text-zinc-500">Voz ElevenLabs</label>
                            <div className="flex gap-2">
                              <div className="flex-1 flex flex-col gap-2">
                                <div className="flex items-center gap-2 bg-black border border-zinc-800 rounded-lg px-3 py-2">
                                  <Mic2 className="w-3 h-3 text-zinc-500" />
                                  <select 
                                    value={selectedElevenLabsVoice}
                                    onChange={(e) => setSelectedElevenLabsVoice(e.target.value)}
                                    className="bg-transparent text-[10px] font-bold text-white outline-none cursor-pointer w-full"
                                  >
                                    {isFetchingVoices ? (
                                      <option value="">Cargando voces...</option>
                                    ) : elevenLabsVoices.length === 0 ? (
                                      <option value="">Sin voces (Verifica permisos)</option>
                                    ) : (
                                      elevenLabsVoices.map(v => (
                                        <option key={v.id} value={v.id}>{v.name}</option>
                                      ))
                                    )}
                                  </select>
                                </div>
                                
                                {/* Fallback for manual Voice ID if permissions are missing */}
                                {elevenLabsVoices.length === 0 && (
                                  <div className="relative">
                                    <input 
                                      type="text"
                                      value={selectedElevenLabsVoice}
                                      onChange={(e) => setSelectedElevenLabsVoice(e.target.value)}
                                      placeholder="O pega ID de voz manual aquí..."
                                      className="w-full bg-black border border-zinc-800 rounded-lg px-3 py-2 text-[10px] font-bold text-white outline-none focus:border-emerald-500 transition-all"
                                    />
                                  </div>
                                )}
                              </div>
                              <button
                                onClick={fetchElevenLabsVoices}
                                disabled={isFetchingVoices}
                                className="p-2 bg-zinc-800 hover:bg-zinc-700 text-zinc-400 hover:text-white rounded-lg transition-all disabled:opacity-50"
                                title="Actualizar voces"
                              >
                                <RotateCw className={cn("w-4 h-4", isFetchingVoices && "animate-spin")} />
                              </button>
                            </div>
                          </div>
                        </div>
                      ) : (
                        <div className="space-y-2">
                          <label className="text-xs font-bold uppercase tracking-wider text-zinc-500">Voz de Narración (Google)</label>
                          <div className="flex gap-2">
                            <div className="flex-1 flex items-center gap-2 bg-black border border-zinc-800 rounded-lg px-3 py-2">
                              <Mic2 className="w-3 h-3 text-zinc-500" />
                              <select 
                                value={selectedVoice}
                                onChange={(e) => setSelectedVoice(e.target.value)}
                                className="bg-transparent text-[10px] font-bold text-white outline-none cursor-pointer w-full"
                              >
                                <option value="Charon">Voz: Charon (Profunda/Calma)</option>
                                <option value="Fenrir">Voz: Fenrir (Dramática/Grave)</option>
                                <option value="Zephyr">Voz: Zephyr (Natural/Media)</option>
                                <option value="Puck">Voz: Puck (Juvenil/Enérgica)</option>
                                <option value="Kore">Voz: Kore (Femenina/Clara)</option>
                              </select>
                            </div>
                            <button
                              onClick={previewVoice}
                              disabled={isPreviewingVoice}
                              className="p-2 bg-zinc-800 hover:bg-zinc-700 text-zinc-400 hover:text-white rounded-lg transition-all disabled:opacity-50"
                              title="Escuchar muestra de voz"
                            >
                              {isPreviewingVoice ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
                            </button>
                          </div>
                        </div>
                      )}

                      <div className="pt-2 border-t border-zinc-800/50">
                        <label className="text-xs font-bold uppercase tracking-wider text-zinc-500 flex items-center gap-2 mb-2">
                          <Upload className="w-3 h-3 text-emerald-500" /> Audio Personalizado
                        </label>
                        <div className="relative group">
                          <input 
                            type="file" 
                            accept="audio/*"
                            onChange={handleCustomAudioUpload}
                            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                          />
                          <div className="w-full py-3 border border-dashed border-zinc-800 rounded-xl flex items-center justify-center gap-2 group-hover:border-emerald-500/50 transition-all bg-black/40">
                            <Volume2 className="w-4 h-4 text-zinc-600 group-hover:text-emerald-500 transition-colors" />
                            <span className="text-[10px] font-bold text-zinc-500 group-hover:text-zinc-300">
                              {story?.isCustomAudio ? "Audio Subido Correctamente" : "Subir Audio Propio (Sincronizar)"}
                            </span>
                          </div>
                        </div>
                        {story?.isCustomAudio && (
                          <p className="text-[9px] text-emerald-500/70 mt-1 italic">
                            * El video se sincronizará automáticamente con la duración de tu audio.
                          </p>
                        )}
                      </div>
                    </div>
                    
                    <div className="space-y-4">
                      <div className="space-y-2">
                        <label className="text-xs font-bold uppercase tracking-wider text-zinc-500">Galería de Estilos</label>
                        <div className="grid grid-cols-2 gap-2">
                          {VISUAL_STYLES.map((style) => (
                            <button
                              key={style}
                              onClick={() => setVisualStyle(style)}
                              className={cn(
                                "py-2 rounded-lg text-[10px] font-bold border transition-all truncate px-1",
                                visualStyle === style ? "bg-emerald-500 border-emerald-500 text-black" : "bg-black border-zinc-800 text-zinc-400 hover:border-zinc-600"
                              )}
                            >
                              {style}
                            </button>
                          ))}
                        </div>
                      </div>

                      <div className="space-y-2">
                        <label className="text-xs font-bold uppercase tracking-wider text-zinc-500">Formato / Red Social</label>
                        <div className="grid grid-cols-2 gap-2">
                          {(["16:9", "9:16", "1:1", "4:3"] as VideoAspectRatio[]).map((ratio) => (
                            <button
                              key={ratio}
                              onClick={() => setVideoAspectRatio(ratio)}
                              className={cn(
                                "py-2 rounded-lg text-[10px] font-bold border transition-all",
                                videoAspectRatio === ratio ? "bg-emerald-500 border-emerald-500 text-black" : "bg-black border-zinc-800 text-zinc-400 hover:border-zinc-600"
                              )}
                            >
                              {ratio === "16:9" ? "YouTube (16:9)" : 
                               ratio === "9:16" ? "TikTok/Reels (9:16)" : 
                               ratio === "1:1" ? "Instagram (1:1)" : "Facebook (4:3)"}
                            </button>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                <button
                  onClick={() => generateStory(undefined, undefined, sidebarTab)}
                  disabled={isGeneratingStory || !prompt}
                  className={cn(
                    "w-full py-3 font-bold rounded-xl transition-all flex items-center justify-center gap-2 disabled:opacity-50",
                    sidebarTab === 'quote' ? "bg-amber-500 hover:bg-amber-400 text-black" : "bg-white hover:bg-zinc-200 text-black"
                  )}
                >
                  {isGeneratingStory ? <Loader2 className="w-4 h-4 animate-spin" /> : (sidebarTab === 'quote' ? <Zap className="w-4 h-4" /> : <Sparkles className="w-4 h-4" />)}
                  {isGeneratingStory 
                    ? (sidebarTab === 'quote' ? "Generando..." : "Redactando...") 
                    : (sidebarTab === 'quote' ? "Generar Frases" : "Generar Historia")}
                </button>
              </section>
            
          

          {error && (
            <div className="bg-red-500/10 border border-red-500/20 rounded-xl p-4 flex flex-col gap-3 text-red-400 text-sm">
              <div className="flex items-start gap-3">
                <AlertCircle className="w-5 h-5 shrink-0" />
                <p>{error}</p>
              </div>
              <div className="pt-2 border-t border-red-500/20 flex flex-col gap-2">
                <div className="flex items-center justify-between">
                  <span className="text-[10px] text-red-500/60 uppercase font-bold tracking-widest">
                    Estado de API: {detectedKey ? "Clave Detectada (" + getApiKeySource() + ")" : "Ninguna Clave Detectada"}
                  </span>
                  <button 
                    onClick={() => setApiKeySelected(false)}
                    className="text-[10px] font-bold text-red-400 hover:text-red-300 underline underline-offset-2"
                  >
                    Reconfigurar Clave
                  </button>
                </div>
                {!detectedKey && (
                  <div className="text-[9px] text-red-500/40 font-mono break-all">
                    D: W.P.E.A: {String(!!getVal(window, 'process.env.API_KEY'))}, 
                    W.A: {String(!!getVal(window, 'API_KEY'))}, 
                    W.P.E.G: {String(!!getVal(window, 'process.env.GEMINI_API_KEY'))},
                    W.G: {String(!!getVal(window, 'GEMINI_API_KEY'))}
                  </div>
                )}
              </div>
            </div>
          )}

          {story && (
            <section className="bg-zinc-900/50 border border-zinc-800 rounded-2xl p-6 space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-xs font-bold uppercase tracking-wider text-zinc-500">Info de Producción</h3>
                <Download className="w-4 h-4 text-zinc-500 cursor-pointer hover:text-white" onClick={downloadAll} />
              </div>
              <div className="space-y-4">
                <div className="p-4 bg-black rounded-xl border border-zinc-800 space-y-2">
                  <div className="flex items-center gap-2 text-emerald-500">
                    <Mic2 className="w-3 h-3" />
                    <span className="text-[10px] font-bold uppercase">Guion de Narración</span>
                  </div>
                  <p className="text-xs text-zinc-400 line-clamp-4 leading-relaxed">{story.narration}</p>
                </div>
                <div className="flex flex-wrap gap-2">
                  {story.hashtags.map((tag, i) => (
                    <span key={i} className="px-2 py-1 bg-zinc-800 rounded text-[10px] text-zinc-500 flex items-center gap-1">
                      <Hash className="w-2 h-2" /> #{tag.replace(/^#/, '')}
                    </span>
                  ))}
                </div>
              </div>
            </section>
          )}
        </div>

        {/* Right Column: Storyboard */}
        <div className="lg:col-span-8 space-y-8 min-h-screen pb-24">
          {!story ? (
            <div className="h-[600px] border-2 border-dashed border-zinc-800 rounded-3xl flex flex-col items-center justify-center text-center p-12 space-y-4">
              <div className="w-16 h-16 bg-zinc-900 rounded-2xl flex items-center justify-center">
                <BookOpen className="w-8 h-8 text-zinc-700" />
              </div>
              <div className="space-y-2">
                <h3 className="text-xl font-bold text-zinc-400">Aún no hay historia</h3>
                <p className="text-zinc-600 max-w-sm">Escribe un prompt o analiza un video para comenzar tu viaje narrativo.</p>
              </div>
            </div>
          ) : (
            <div className="space-y-8">
              <div className="flex items-end justify-between">
                <div className="space-y-1">
                  <span className="text-[10px] font-bold uppercase tracking-[0.2em] text-emerald-500">Storyboard Visual</span>
                  <h2 className="text-4xl font-bold tracking-tight">{story.title}</h2>
                </div>
                <div className="flex gap-2">
                  {story.segments.map((_, i) => (
                    <button
                      key={i}
                      onClick={() => setActiveSegmentIndex(i)}
                      className={cn(
                        "w-10 h-1 flex rounded-full transition-all",
                        activeSegmentIndex === i ? "bg-emerald-500 w-16" : "bg-zinc-800 hover:bg-zinc-700"
                      )}
                    />
                  ))}
                </div>
              </div>

              <AnimatePresence mode="wait">
                <motion.div
                  key={story.segments[activeSegmentIndex].id}
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  className="space-y-6"
                >
                  {/* Visual Display */}
                  <div className={cn(
                    "relative bg-zinc-900 rounded-3xl overflow-hidden border border-zinc-800 group transition-all duration-500",
                    videoAspectRatio === "16:9" ? "aspect-video" : 
                    videoAspectRatio === "9:16" ? "aspect-[9/16] max-w-[400px] mx-auto" :
                    videoAspectRatio === "1:1" ? "aspect-square max-w-[500px] mx-auto" :
                    "aspect-[4/3] max-w-[600px] mx-auto"
                  )}>
                    {story.segments[activeSegmentIndex].videoUrl && videoProvider !== "CinematicPan" ? (
                      <div className="relative w-full h-full overflow-hidden">
                        <video src={story.segments[activeSegmentIndex].videoUrl} controls autoPlay loop className="w-full h-full object-cover" />
                        
                        {/* UI Text Overlay (Only for Quotes if not burned into image) */}
                        {!burnTextIntoImages && story.type === 'quote' && story.segments[activeSegmentIndex].text && (
                          <div className="absolute inset-0 p-8 pointer-events-none flex items-start justify-center pt-12">
                            <motion.div 
                              initial={{ opacity: 0, y: -20 }}
                              animate={{ opacity: 1, y: 0 }}
                              className="max-w-[85%] shadow-2xl bg-black/40 backdrop-blur-sm px-8 py-6 rounded-3xl border border-white/10"
                            >
                              <p 
                                className="text-white text-center leading-relaxed tracking-wide text-2xl md:text-3xl font-serif italic"
                                style={{ fontFamily: 'var(--font-serif)' }}
                              >
                                {story.segments[activeSegmentIndex].text}
                              </p>
                            </motion.div>
                          </div>
                        )}
                      </div>
                    ) : (story.segments[activeSegmentIndex].imageUrl || story.segments[activeSegmentIndex].videoUrl) ? (
                      <div className="relative w-full h-full overflow-hidden">
                        <img 
                          src={story.segments[activeSegmentIndex].imageUrl || story.segments[activeSegmentIndex].videoUrl} 
                          className={cn(
                            "w-full h-full object-cover",
                            videoProvider === "CinematicPan" ? "animate-cinematic-pan" : ""
                          )}
                          style={videoProvider === "CinematicPan" ? { 
                            animationDuration: `${(story.segments[activeSegmentIndex].duration || 5) * 2}s` 
                          } : {}}
                          alt="Segment visual"
                          referrerPolicy="no-referrer"
                        />
                        
                        {/* UI Text Overlay (Only for Quotes if not burned into image) */}
                        {!burnTextIntoImages && story.type === 'quote' && story.segments[activeSegmentIndex].text && (
                          <div className="absolute inset-0 p-8 pointer-events-none flex items-start justify-center pt-12">
                            <motion.div 
                              initial={{ opacity: 0, y: -20 }}
                              animate={{ opacity: 1, y: 0 }}
                              className="max-w-[85%] shadow-2xl bg-black/40 backdrop-blur-sm px-8 py-6 rounded-3xl border border-white/10"
                            >
                              <p 
                                className="text-white text-center leading-relaxed tracking-wide text-2xl md:text-3xl font-serif italic"
                                style={{ fontFamily: 'var(--font-serif)' }}
                              >
                                {story.segments[activeSegmentIndex].text}
                              </p>
                            </motion.div>
                          </div>
                        )}
                      </div>
                    ) : (
                      <div className="absolute inset-0 flex flex-col items-center justify-center space-y-4">
                        {story.segments[activeSegmentIndex].isGeneratingImage || story.segments[activeSegmentIndex].isGeneratingVideo ? (
                          <div className="flex flex-col items-center gap-4">
                            <div className="relative">
                              <Loader2 className="w-12 h-12 text-emerald-500 animate-spin" />
                              <div className="absolute inset-0 blur-xl bg-emerald-500/20 animate-pulse" />
                            </div>
                            <span className="text-sm font-medium text-zinc-400 animate-pulse">
                              {story.segments[activeSegmentIndex].isGeneratingVideo 
                                ? (story.segments[activeSegmentIndex].imageUrl ? "Animando Imagen Pro (Veo 3.1)..." : "Sintetizando Video Pro (Veo 3.1 HQ)...") 
                                : "Visualizando Escena..."}
                            </span>
                          </div>
                        ) : (
                          <>
                            <ImageIcon className="w-12 h-12 text-zinc-800" />
                            <span className="text-sm text-zinc-600">Visual aún no generada</span>
                          </>
                        )}
                      </div>
                    )}

                    {/* Action Overlay */}
                    <div className="absolute bottom-6 right-6 flex gap-3 opacity-0 group-hover:opacity-100 transition-opacity">
                      <button 
                        onClick={() => generateImage(activeSegmentIndex)}
                        disabled={story.segments[activeSegmentIndex].isGeneratingImage}
                        className="p-3 bg-black/80 backdrop-blur-md border border-white/10 rounded-xl hover:bg-white hover:text-black transition-all disabled:opacity-50"
                      >
                        <ImageIcon className="w-5 h-5" />
                      </button>
                      <button 
                        onClick={() => generateVideo(activeSegmentIndex)}
                        disabled={story.segments[activeSegmentIndex].isGeneratingVideo}
                        className="p-3 bg-black/80 backdrop-blur-md border border-white/10 rounded-xl hover:bg-white hover:text-black transition-all disabled:opacity-50"
                      >
                        <Video className="w-5 h-5" />
                      </button>
                    </div>
                  </div>

                  {/* Text Content */}
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                    <div className="md:col-span-2 space-y-4">
                      <div className="flex items-center gap-3">
                        <div className="flex items-center gap-1.5 px-3 py-1 bg-emerald-500/10 border border-emerald-500/20 rounded-full">
                          <span className="w-1.5 h-1.5 bg-emerald-500 rounded-full animate-pulse" />
                          <span className="text-emerald-500 text-[10px] font-bold uppercase tracking-widest">
                            Segmento {activeSegmentIndex + 1}
                          </span>
                        </div>
                        {story.segments[activeSegmentIndex].cameraAngle && (
                          <div className="flex items-center gap-1.5 px-3 py-1 bg-zinc-800/50 backdrop-blur-md border border-zinc-700/50 rounded-full">
                            <Video className="w-3 h-3 text-zinc-400" />
                            <span className="text-zinc-300 text-[10px] font-bold uppercase tracking-widest">
                              {story.segments[activeSegmentIndex].cameraAngle === "Standard" ? "Estándar" :
                               story.segments[activeSegmentIndex].cameraAngle === "Close-up" ? "Primer Plano" :
                               story.segments[activeSegmentIndex].cameraAngle === "Extreme Close-up" ? "Primerísimo Primer Plano" :
                               story.segments[activeSegmentIndex].cameraAngle === "Conversation" ? "Conversación" :
                               story.segments[activeSegmentIndex].cameraAngle === "Aerial" ? "Aérea" :
                               story.segments[activeSegmentIndex].cameraAngle === "Wide Shot" ? "Plano General" :
                               story.segments[activeSegmentIndex].cameraAngle === "Fisheye" ? "Ojo de Pez" :
                               story.segments[activeSegmentIndex].cameraAngle === "Low Angle" ? "Contrapicado" :
                               story.segments[activeSegmentIndex].cameraAngle === "Dutch Angle" ? "Plano Holandés" :
                               story.segments[activeSegmentIndex].cameraAngle === "Mirror Reflection" ? "Reflejo en Espejo" :
                               story.segments[activeSegmentIndex].cameraAngle}
                            </span>
                          </div>
                        )}
                        <div className="h-px flex-1 bg-gradient-to-r from-zinc-800 to-transparent" />
                      </div>
                      <div className="prose prose-invert max-w-none text-lg leading-relaxed text-zinc-300">
                        <ReactMarkdown>{story.segments[activeSegmentIndex].text}</ReactMarkdown>
                      </div>
                    </div>

                    <div className="space-y-4">
                      <div className="p-5 bg-black/40 backdrop-blur-xl border border-white/5 rounded-3xl space-y-4 shadow-2xl">
                        <div className="flex items-center justify-between">
                          <h4 className="text-[10px] font-bold uppercase tracking-widest text-zinc-500">Prompt Visual</h4>
                          <Zap className="w-3 h-3 text-emerald-500/50" />
                        </div>
                        <div className="relative group/prompt">
                          <p className="text-[11px] text-zinc-400 leading-relaxed italic font-serif">
                            "{story.segments[activeSegmentIndex].imagePrompt}"
                          </p>
                          <div className="absolute -inset-2 bg-emerald-500/5 rounded-lg opacity-0 group-hover/prompt:opacity-100 transition-opacity pointer-events-none" />
                        </div>
                      </div>

                      {story.segments[activeSegmentIndex].videoDescription && (
                        <div className="p-5 bg-emerald-500/5 backdrop-blur-xl border border-emerald-500/10 rounded-3xl space-y-3 shadow-2xl">
                          <div className="flex items-center justify-between">
                            <h4 className="text-[10px] font-bold uppercase tracking-widest text-emerald-500/70">Movimiento y Escena</h4>
                            <Video className="w-3 h-3 text-emerald-500/50" />
                          </div>
                          <p className="text-[11px] text-zinc-300 leading-relaxed font-medium">
                            {story.segments[activeSegmentIndex].videoDescription}
                          </p>
                        </div>
                      )}
                      
                      <div className="flex gap-3">
                        <button 
                          onClick={() => setActiveSegmentIndex(prev => Math.max(0, prev - 1))}
                          disabled={activeSegmentIndex === 0}
                          className="flex-1 py-4 bg-zinc-900/50 hover:bg-zinc-800/50 backdrop-blur-md border border-white/5 rounded-2xl text-[10px] font-bold uppercase tracking-widest disabled:opacity-20 transition-all hover:scale-[1.02] active:scale-[0.98]"
                        >
                          Anterior
                        </button>
                        <button 
                          onClick={() => setActiveSegmentIndex(prev => Math.min(story.segments.length - 1, prev + 1))}
                          disabled={activeSegmentIndex === story.segments.length - 1}
                          className="flex-1 py-4 bg-white text-black rounded-2xl text-[10px] font-bold uppercase tracking-widest disabled:opacity-20 transition-all hover:scale-[1.02] active:scale-[0.98] shadow-lg shadow-white/10"
                        >
                          Siguiente
                        </button>
                      </div>

                      {story.segments[activeSegmentIndex].error && (
                        <div className="p-3 bg-red-500/10 border border-red-500/20 rounded-xl space-y-2">
                          <p className="text-[10px] text-red-400 font-medium">{story.segments[activeSegmentIndex].error}</p>
                          <div className="flex gap-2">
                            <button 
                              onClick={() => generateImage(activeSegmentIndex)}
                              className="flex-1 py-1.5 bg-red-500/20 hover:bg-red-500/30 text-red-400 text-[10px] font-bold rounded-lg transition-colors border border-red-500/30"
                            >
                              Reintentar Imagen
                            </button>
                            <button 
                              onClick={() => generateVideo(activeSegmentIndex)}
                              className="flex-1 py-1.5 bg-red-500/20 hover:bg-red-500/30 text-red-400 text-[10px] font-bold rounded-lg transition-colors border border-red-500/30"
                            >
                              Reintentar Video
                            </button>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </motion.div>
              </AnimatePresence>

              {/* Sequential Player Section */}
              <section className="bg-zinc-900/40 border border-zinc-800/50 rounded-3xl overflow-hidden shadow-2xl">
                <div className="p-6 border-b border-zinc-800/50 bg-zinc-900/20">
                  <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
                    <div className="flex items-center gap-4">
                      <div className="w-12 h-12 rounded-2xl bg-emerald-500/10 flex items-center justify-center border border-emerald-500/20">
                        <Layers className="w-6 h-6 text-emerald-500" />
                      </div>
                      <div>
                        <h3 className="font-bold text-xl tracking-tight">Secuencia Completa</h3>
                        <p className="text-xs text-zinc-500 mt-0.5">Gestión de activos y exportación final</p>
                      </div>
                      {(isBulkGeneratingImages || isBulkGeneratingVideos || isBulkProcessing) && (
                        <div className="flex items-center gap-2 px-3 py-1.5 bg-emerald-500/10 border border-emerald-500/20 rounded-full animate-pulse">
                          <Loader2 className="w-3 h-3 text-emerald-500 animate-spin" />
                          <span className="text-[10px] font-bold text-emerald-500 uppercase tracking-widest">
                            {isBulkProcessing ? "Procesando Todo..." : isBulkGeneratingImages ? "Generando Imágenes..." : "Animando Escenas..."}
                          </span>
                        </div>
                      )}
                    </div>

                    <div className="flex items-center gap-4 bg-black/40 p-3 rounded-2xl border border-white/5 backdrop-blur-sm">
                      <div className="flex flex-col gap-1">
                        <span className={cn(
                          "text-[9px] font-bold uppercase tracking-widest",
                          (story?.audioUrl && story?.segments.every(s => s.audioUrl)) ? "text-emerald-500" : "text-zinc-500"
                        )}>
                          {isGeneratingAudio ? (
                            <div className="flex items-center gap-2">
                              <Loader2 className="w-2.5 h-2.5 animate-spin" />
                              <span>Generando {audioProgress?.current}/{audioProgress?.total}...</span>
                            </div>
                          ) : (story?.audioUrl && story?.segments.every(s => s.audioUrl)) ? "Narración Completa" : "Narración Incompleta"}
                        </span>
                        {story?.audioUrl ? (
                          <audio src={story.audioUrl} controls className="h-7 w-48 md:w-56" />
                        ) : (
                          <p className="text-[10px] text-zinc-500 italic px-2">Genera la narración para habilitar el audio</p>
                        )}
                      </div>
                    </div>
                  </div>
                </div>

                <div className="p-6 space-y-6">
                  <div className="flex flex-col gap-4">
                    <div className="flex items-center justify-between">
                      <div className="flex flex-col gap-2">
                        <h4 className="text-[10px] font-bold text-zinc-500 uppercase tracking-[0.2em]">Acciones de Procesamiento</h4>
                      </div>
                      {bulkProgress && (
                        <div className="h-1 w-32 bg-zinc-800 rounded-full overflow-hidden">
                          <div 
                            className="h-full bg-emerald-500 transition-all duration-500" 
                            style={{ width: `${(bulkProgress.current / bulkProgress.total) * 100}%` }}
                          />
                        </div>
                      )}
                    </div>
                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
                      <button
                        onClick={generateAudio}
                        disabled={isGeneratingAudio || isBulkGeneratingImages || isBulkGeneratingVideos}
                        className="group px-4 py-3 bg-zinc-800/40 hover:bg-zinc-800/60 text-white text-xs font-bold rounded-xl flex items-center justify-center gap-3 transition-all disabled:opacity-50 border border-zinc-700/20"
                      >
                        {isGeneratingAudio ? <Loader2 className="w-4 h-4 animate-spin text-emerald-500" /> : <Mic2 className="w-4 h-4 text-zinc-400 group-hover:text-white transition-colors" />}
                        <span>Narración</span>
                      </button>

                      <button
                        onClick={() => generateAllImages()}
                        disabled={isBulkGeneratingImages}
                        className="group px-4 py-3 bg-zinc-800/40 hover:bg-zinc-800/60 text-white text-xs font-bold rounded-xl flex items-center justify-center gap-3 transition-all disabled:opacity-50 border border-zinc-700/20"
                      >
                        {isBulkGeneratingImages ? (
                          <>
                            <Loader2 className="w-4 h-4 animate-spin text-emerald-500" />
                            <span className="text-emerald-400">{bulkProgress ? `${bulkProgress.current}/${bulkProgress.total}` : "Generando..."}</span>
                          </>
                        ) : (
                          <>
                            <ImageIcon className="w-4 h-4 text-zinc-400 group-hover:text-white transition-colors" />
                            <span>Imágenes</span>
                          </>
                        )}
                      </button>

                      <button
                        onClick={() => generateAllVideos()}
                        disabled={isBulkGeneratingVideos}
                        className="group px-4 py-3 bg-zinc-800/40 hover:bg-zinc-800/60 text-white text-xs font-bold rounded-xl flex items-center justify-center gap-3 transition-all disabled:opacity-50 border border-zinc-700/20"
                      >
                        {isBulkGeneratingVideos ? (
                          <>
                            <Loader2 className="w-4 h-4 animate-spin text-emerald-500" />
                            <span className="text-emerald-400">{bulkProgress ? `${bulkProgress.current}/${bulkProgress.total}` : "Animando..."}</span>
                          </>
                        ) : (
                          <>
                            <Video className="w-4 h-4 text-zinc-400 group-hover:text-white transition-colors" />
                            <span>Videos</span>
                          </>
                        )}
                      </button>

                      <button
                        onClick={retryAllFailed}
                        disabled={isBulkProcessing || !story?.segments.some(s => s.error)}
                        className="group px-4 py-3 bg-red-500/10 hover:bg-red-500/20 text-red-400 text-xs font-bold rounded-xl flex items-center justify-center gap-3 transition-all border border-red-500/20 disabled:opacity-50"
                      >
                        {isBulkProcessing && bulkMode === "all" ? (
                          <>
                            <Loader2 className="w-4 h-4 animate-spin" />
                            <span>{bulkProgress ? `${bulkProgress.current}/${bulkProgress.total}` : "..."}</span>
                          </>
                        ) : (
                          <>
                            <RefreshCw className="w-4 h-4" />
                            <span>Reintentar Fallidos</span>
                          </>
                        )}
                      </button>

                      <button
                        onClick={processEverything}
                        disabled={isBulkProcessing}
                        className={cn(
                          "group px-4 py-3 text-xs font-bold rounded-xl flex items-center justify-center gap-3 transition-all disabled:opacity-50 border shadow-lg",
                          !isBulkProcessing && story?.segments.some(s => !s.imageUrl) 
                            ? "bg-emerald-500 text-black border-emerald-400 animate-pulse shadow-emerald-500/40" 
                            : "bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-400 border-emerald-500/20 shadow-emerald-500/5"
                        )}
                      >
                        {isBulkProcessing ? (
                          <>
                            <Loader2 className="w-4 h-4 animate-spin" />
                            <span>{bulkProgress ? `${bulkProgress.current}/${bulkProgress.total}` : "..."}</span>
                          </>
                        ) : (
                          <>
                            <Sparkles className={cn("w-4 h-4 group-hover:rotate-12 transition-transform", !isBulkProcessing && story?.segments.some(s => !s.imageUrl) ? "animate-bounce" : "")} />
                            <span>Auto</span>
                          </>
                        )}
                      </button>
                    </div>
                  </div>

                  <div className="flex flex-col md:flex-row items-center justify-between gap-6 pt-6 border-t border-zinc-800/50">
                    <div className="flex flex-col gap-2">
                      <h4 className="text-[10px] font-bold text-zinc-500 uppercase tracking-[0.2em]">Finalización</h4>
                      <div className="flex items-center gap-3">
                        <button 
                          onClick={clearAllAssets}
                          className="px-4 py-2.5 bg-red-500/5 hover:bg-red-500/10 text-red-500 text-[10px] font-bold rounded-xl flex items-center gap-2 transition-all border border-red-500/10"
                        >
                          <X className="w-3.5 h-3.5" /> LIMPIAR ASSETS
                        </button>
                        <p className="text-[10px] text-zinc-600 italic max-w-[300px] leading-tight">
                          * Los audios y videos se guardan solo en esta sesión. Si recargas la página, deberás generarlos de nuevo.
                        </p>
                      </div>
                    </div>

                    <button 
                      onClick={() => setShowPackagePreview(true)}
                      className="w-full md:w-auto px-6 py-4 bg-zinc-900 border border-zinc-800 hover:border-emerald-500/50 text-white text-sm font-black rounded-2xl flex items-center justify-center gap-3 transition-all group"
                    >
                      <Download className="w-5 h-5 group-hover:-translate-y-0.5 transition-transform" /> 
                      <span>DESCARGAR ASSETS</span>
                    </button>

                    <button 
                      onClick={async () => {
                        const isAudioComplete = story?.audioUrl && story?.segments.every(s => s.audioUrl);
                        if (!isAudioComplete) {
                          const confirmed = await requestConfirm({
                            title: "Audio Incompleto",
                            message: "Faltan audios en algunos segmentos. El video final podría estar en silencio en esas partes. ¿Deseas continuar?",
                            confirmText: "Continuar de todos modos"
                          });
                          if (confirmed) setIsPreviewingFinal(true);
                        } else {
                          setIsPreviewingFinal(true);
                        }
                      }}
                      className="w-full md:w-auto px-10 py-4 bg-emerald-500 hover:bg-emerald-400 text-black text-sm font-black rounded-2xl flex items-center justify-center gap-3 transition-all shadow-2xl shadow-emerald-500/20 hover:scale-[1.02] active:scale-[0.98] group"
                    >
                      <Zap className="w-5 h-5 group-hover:scale-110 transition-transform" /> 
                      <span>GENERAR VIDEO FINAL (AUTOMÁTICO)</span>
                    </button>
                  </div>
                </div>

                  <div className="flex items-center justify-between px-4 mb-3">
                    <h3 className="text-xs font-black text-white uppercase tracking-tight flex items-center gap-2">
                      <LayoutGrid className="w-3.5 h-3.5 text-emerald-500" />
                      Segmentos de la Historia
                    </h3>
                    <div className="flex items-center gap-2">
                      <button 
                        onClick={() => setIsSyncMode(!isSyncMode)}
                        className={cn(
                          "px-2 py-1 text-[9px] font-bold rounded-lg border flex items-center gap-1.5 transition-all",
                          isSyncMode 
                            ? "bg-emerald-500 text-black border-emerald-500" 
                            : "bg-zinc-800/50 text-zinc-400 border-zinc-700/50 hover:text-white"
                        )}
                      >
                        <Clock className="w-2.5 h-2.5" />
                        {isSyncMode ? "MODO SYNC ON" : "AJUSTAR TIEMPOS"}
                      </button>
                      {isSyncMode && (
                        <button 
                          onClick={async () => {
                            const confirmed = await requestConfirm({
                              title: "Reiniciar Tiempos",
                              message: "¿Estás seguro de que deseas reiniciar todos los tiempos a la estimación automática basada en el texto?",
                              confirmText: "Reiniciar"
                            });
                            if (!confirmed) return;
                            
                            setStory(prev => {
                              if (!prev) return null;
                              const duration = prev.audioBlob ? prev.audioBlob.size / (24000 * 2) : (prev.segments.length * 5); // Fallback estimate
                              
                              const segmentMetrics = prev.segments.map(s => {
                                const charCount = s.text.length;
                                const wordCount = s.text.split(/\s+/).filter(w => w.length > 0).length;
                                const sentenceCount = (s.text.match(/[.!?]+/g) || []).length;
                                return (charCount * 0.5) + (wordCount * 2.8) + (sentenceCount * 1.5); 
                              });

                              const totalWeight = segmentMetrics.reduce((sum, w) => sum + w, 0);
                              let currentOffset = 0;
                              const updatedSegments = prev.segments.map((s, idx) => {
                                const proportion = segmentMetrics[idx] / (totalWeight || 1);
                                const segDuration = proportion * (prev.audioUrl ? duration : prev.segments.length * 5);
                                const start = currentOffset;
                                currentOffset += segDuration;
                                
                                return {
                                  ...s,
                                  duration: segDuration,
                                  audioStart: start,
                                  audioEnd: currentOffset
                                };
                              });
                              return { ...prev, segments: updatedSegments };
                            });
                          }}
                          className="px-2 py-1 bg-red-500/10 hover:bg-red-500/20 text-red-500 text-[9px] font-bold rounded-lg border border-red-500/20 flex items-center gap-1.5 transition-all"
                        >
                          <RefreshCw className="w-2.5 h-2.5" />
                          REINICIAR
                        </button>
                      )}
                      {story.narration && (
                        <>
                        <button 
                          onClick={() => {
                            navigator.clipboard.writeText(story.narration);
                          }}
                          className="px-2 py-1 bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-500 text-[9px] font-bold rounded-lg border border-emerald-500/20 flex items-center gap-1.5 transition-all"
                        >
                          <Copy className="w-2.5 h-2.5" />
                          COPIAR NARRACIÓN
                        </button>
                        <button 
                          onClick={downloadPromptsCSV}
                          className="px-2 py-1 bg-sky-500/10 hover:bg-sky-500/20 text-sky-500 text-[9px] font-bold rounded-lg border border-sky-500/20 flex items-center gap-1.5 transition-all"
                        >
                          <FileText className="w-2.5 h-2.5" />
                          DESCARGAR CSV PROMPTS
                        </button>
                        <button 
                          onClick={downloadVideoDescriptionsCSV}
                          className="px-2 py-1 bg-sky-500/10 hover:bg-sky-500/20 text-sky-500 text-[9px] font-bold rounded-lg border border-sky-500/20 flex items-center gap-1.5 transition-all"
                        >
                          <FileText className="w-2.5 h-2.5" />
                          DESCARGAR MOVIMIENTOS
                        </button>
                        </>
                      )}
                    </div>
                  </div>
                
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3 px-4 pb-4">
                  {story.segments.map((seg, i) => (
                    <div key={i} className="space-y-2 group/item">
                      <div className={cn(
                        "bg-black rounded-xl border border-zinc-800 overflow-hidden relative group transition-all",
                        videoAspectRatio === "16:9" ? "aspect-video" : "aspect-[9/16]"
                      )}>
                        {seg.videoUrl ? (
                          <div className="relative w-full h-full">
                            <video src={seg.videoUrl} className="w-full h-full object-cover" />
                            
                            {/* Mini UI Text Overlay (Only for Quotes) */}
                            {!burnTextIntoImages && story.type === 'quote' && seg.text && (
                              <div className="absolute inset-0 p-2 pointer-events-none flex items-start justify-center pt-2">
                                <div className="bg-black/40 backdrop-blur-[2px] px-2 py-1.5 rounded-lg border border-white/5 max-w-[90%] shadow-lg">
                                  <p 
                                    className="text-white text-center line-clamp-2 leading-tight text-[7px] font-serif italic"
                                    style={{ fontFamily: 'var(--font-serif)' }}
                                  >
                                    {seg.text}
                                  </p>
                                </div>
                              </div>
                            )}
                          </div>
                        ) : seg.isGeneratingVideo ? (
                          <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/60 backdrop-blur-sm">
                            <Loader2 className="w-6 h-6 animate-spin text-emerald-500 mb-2" />
                            <span className="text-[8px] font-bold text-emerald-500 uppercase tracking-widest">Animando...</span>
                          </div>
                        ) : seg.isGeneratingImage ? (
                          <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/60 backdrop-blur-sm">
                            <Loader2 className="w-6 h-6 animate-spin text-emerald-500 mb-2" />
                            <span className="text-[8px] font-bold text-emerald-500 uppercase tracking-widest">Generando...</span>
                          </div>
                        ) : seg.imageUrl ? (
                          <div className="relative w-full h-full overflow-hidden">
                            <img 
                              src={seg.imageUrl} 
                              className={cn(
                                "w-full h-full object-cover",
                                videoProvider === "CinematicPan" ? "animate-cinematic-pan" : ""
                              )} 
                              style={videoProvider === "CinematicPan" ? { 
                                animationDuration: `${(seg.duration || 5) * 2}s` 
                              } : {}}
                              alt="Segment visual"
                              referrerPolicy="no-referrer"
                            />
                            
                            {/* Mini UI Text Overlay (Only for Quotes) */}
                            {!burnTextIntoImages && story.type === 'quote' && seg.text && (
                              <div className="absolute inset-0 p-2 pointer-events-none flex items-start justify-center pt-2">
                                <div className="bg-black/40 backdrop-blur-[2px] px-2 py-1.5 rounded-lg border border-white/5 max-w-[90%] shadow-lg">
                                  <p 
                                    className="text-white text-center line-clamp-2 leading-tight text-[7px] font-serif italic"
                                    style={{ fontFamily: 'var(--font-serif)' }}
                                  >
                                    {seg.text}
                                  </p>
                                </div>
                              </div>
                            )}

                            {seg.error && (
                              <div className="absolute inset-0 bg-black/60 backdrop-blur-[2px] flex flex-col items-center justify-center p-2 text-center">
                                <AlertCircle className="w-5 h-5 text-red-500 mb-1" />
                                <span className="text-[8px] font-bold text-red-500 uppercase tracking-widest mb-1">Error de Video</span>
                                <p className="text-[7px] text-zinc-300 leading-tight mb-2 line-clamp-2 px-1">{seg.error}</p>
                                <button 
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    generateVideo(i);
                                  }}
                                  className="px-3 py-1 bg-red-500 text-white text-[8px] font-bold rounded-lg hover:bg-red-600 transition-colors shadow-lg shadow-red-500/20"
                                >
                                  REINTENTAR
                                </button>
                              </div>
                            )}

                            {/* Individual Download Button */}
                            <div className="absolute top-2 right-2 flex gap-2 opacity-0 group-hover/item:opacity-100 transition-opacity">
                              <button 
                                onClick={async (e) => {
                                  e.stopPropagation();
                                  const url = seg.videoUrl || seg.imageUrl;
                                  if (!url) return;
                                  const isCinematicPan = seg.videoUrl && seg.videoUrl === seg.imageUrl;
                                  const isRealVideo = seg.videoUrl && !isCinematicPan;
                                  
                                  try {
                                    const response = await fetch(url);
                                    const blob = await response.blob();
                                    const blobUrl = URL.createObjectURL(blob);
                                    const link = document.createElement('a');
                                    link.href = blobUrl;
                                    link.download = isRealVideo ? `segment_${i + 1}_video.mp4` : `segment_${i + 1}_image.png`;
                                    document.body.appendChild(link);
                                    link.click();
                                    document.body.removeChild(link);
                                    URL.revokeObjectURL(blobUrl);
                                  } catch (err) {
                                    console.error("Download error:", err);
                                    const link = document.createElement('a');
                                    link.href = url;
                                    link.download = isRealVideo ? `segment_${i + 1}_video.mp4` : `segment_${i + 1}_image.png`;
                                    link.click();
                                  }
                                }}
                                className="p-1.5 bg-black/60 backdrop-blur-md border border-white/10 rounded-lg hover:bg-emerald-500 hover:text-black transition-all"
                                title="Descargar"
                              >
                                <Download className="w-3 h-3" />
                              </button>
                            </div>
                          </div>
                        ) : seg.error ? (
                          <div className="absolute inset-0 flex flex-col items-center justify-center bg-red-500/10 backdrop-blur-sm p-2 text-center">
                            <AlertCircle className="w-5 h-5 text-red-500 mb-1" />
                            <span className="text-[8px] font-bold text-red-500 uppercase tracking-widest mb-1">Error</span>
                            <p className="text-[7px] text-red-400/80 leading-tight mb-2 line-clamp-2 px-1">{seg.error}</p>
                            <div className="flex flex-col gap-1.5">
                              <button 
                                onClick={() => {
                                  if (seg.error?.toLowerCase().includes("imagen") || !seg.imageUrl) {
                                    generateImage(i);
                                  } else {
                                    generateVideo(i);
                                  }
                                }}
                                className="px-3 py-1 bg-red-500 text-white text-[8px] font-bold rounded-lg hover:bg-red-600 transition-colors shadow-lg shadow-red-500/20"
                              >
                                REINTENTAR
                              </button>
                              {(seg.error?.toLowerCase().includes("google") || seg.error?.toLowerCase().includes("saturado")) && (
                                <button 
                                  onClick={() => {
                                    setImageProvider("Pollinations");
                                    setTimeout(() => generateImage(i), 100);
                                  }}
                                  className="px-3 py-1 bg-zinc-800 text-zinc-300 text-[8px] font-bold rounded-lg hover:bg-zinc-700 transition-colors border border-zinc-700"
                                >
                                  USAR POLLINATIONS
                                </button>
                              )}
                            </div>
                          </div>
                        ) : (
                          <div className="absolute inset-0 flex items-center justify-center">
                            <ImageIcon className="w-4 h-4 text-zinc-800" />
                          </div>
                        )}
                        
                        {!seg.isGeneratingVideo && !seg.error && (
                          <div className="absolute inset-0 bg-black/40 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                            <button onClick={() => setActiveSegmentIndex(i)} className="p-2 bg-white text-black rounded-full">
                              <Play className="w-3 h-3 fill-current" />
                            </button>
                          </div>
                        )}
                      </div>
                      <div className="space-y-1.5">
                        <div className="flex items-center justify-between px-1">
                          <span className="text-[10px] font-bold text-zinc-500">Parte {i + 1}</span>
                          <div className="flex items-center gap-2">
                            {isSyncMode && (
                              <div className="flex items-center gap-1 bg-zinc-900 border border-zinc-800 rounded-md px-1.5 py-0.5">
                                <Clock className="w-2 h-2 text-zinc-500" />
                                <input 
                                  type="number"
                                  step="0.1"
                                  min="0.1"
                                  value={seg.duration?.toFixed(1) || "5.0"}
                                  onChange={(e) => {
                                    const newDuration = parseFloat(e.target.value);
                                    if (isNaN(newDuration)) return;
                                    setStory(prev => {
                                      if (!prev) return null;
                                      const newSegments = [...prev.segments];
                                      newSegments[i] = { ...newSegments[i], duration: newDuration };
                                      
                                      // Recalcular offsets de audio si estamos en modo manual
                                      let offset = 0;
                                      const updated = newSegments.map(s => {
                                        const start = offset;
                                        offset += s.duration || 5;
                                        return { ...s, audioStart: start, audioEnd: offset };
                                      });
                                      
                                      return { ...prev, segments: updated };
                                    });
                                  }}
                                  className={cn(
                                    "w-8 bg-transparent text-[9px] font-mono focus:outline-none text-center",
                                    (seg.duration || 0) > 8.5 ? "text-red-500 font-bold" : "text-emerald-500"
                                  )}
                                />
                                <span className={cn(
                                  "text-[8px] font-bold",
                                  (seg.duration || 0) > 8.5 ? "text-red-500" : "text-zinc-600"
                                )}>s</span>
                                {(seg.duration || 0) > 8.5 && (
                                  <div title="Segmento muy largo (puede sentirse lento)">
                                    <AlertTriangle className="w-2 h-2 text-red-500 animate-pulse" />
                                  </div>
                                )}
                              </div>
                            )}
                            <button 
                              onClick={() => navigator.clipboard.writeText(seg.text)}
                              className="p-1 text-zinc-600 hover:text-emerald-500 transition-colors opacity-0 group-hover/item:opacity-100"
                              title="Copiar frase"
                            >
                              <Copy className="w-2.5 h-2.5" />
                            </button>
                            {seg.videoUrl && (
                              <button 
                                onClick={(e) => {
                                  e.stopPropagation();
                                  const blob = seg.videoBlob;
                                  const url = blob ? URL.createObjectURL(blob) : seg.videoUrl!;
                                  const link = document.createElement('a');
                                  link.href = url;
                                  link.download = `segmento_${i + 1}.mp4`;
                                  link.click();
                                  if (blob) setTimeout(() => URL.revokeObjectURL(url), 100);
                                }}
                                className="p-1 text-zinc-600 hover:text-emerald-500 transition-colors opacity-0 group-hover/item:opacity-100"
                                title="Descargar Video"
                              >
                                <Download className="w-2.5 h-2.5" />
                              </button>
                            )}
                            {seg.imageUrl && !seg.videoUrl && (
                              <button 
                                onClick={(e) => {
                                  e.stopPropagation();
                                  const link = document.createElement('a');
                                  link.href = seg.imageUrl!;
                                  link.download = `segmento_${i + 1}.png`;
                                  link.click();
                                }}
                                className="p-1 text-zinc-600 hover:text-emerald-500 transition-colors opacity-0 group-hover/item:opacity-100"
                                title="Descargar Imagen"
                              >
                                <Download className="w-2.5 h-2.5" />
                              </button>
                            )}
                            {seg.videoUrl ? (
                              <CheckCircle2 className="w-3 h-3 text-emerald-500" />
                            ) : seg.isGeneratingVideo ? (
                              <Loader2 className="w-3 h-3 animate-spin text-emerald-500" />
                            ) : seg.error ? (
                              <AlertCircle className="w-3 h-3 text-red-500" />
                            ) : (
                              <div className="w-3 h-3 rounded-full border border-zinc-800" />
                            )}
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
                <p className="text-[10px] text-zinc-600 italic px-6 pb-6 text-center border-t border-zinc-800/30 pt-4">
                  Consejo: Descarga el paquete para obtener todos los recursos en alta resolución, el guion de narración y los metadatos.
                </p>
              </section>
              {/* Segment Thumbnails - Improved Layout to prevent "getting lost" */}
              <div className="space-y-4 pt-8 border-t border-zinc-800">
                <div className="flex items-center justify-between">
                  <h3 className="text-[10px] font-bold uppercase tracking-[0.2em] text-zinc-500">Segmentos de la Historia</h3>
                  <span className="text-[10px] text-zinc-600 font-mono">{story.segments.length} Escenas</span>
                </div>
                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
                  {story.segments.map((seg, i) => (
                    <div 
                      key={seg.id}
                      onClick={() => setActiveSegmentIndex(i)}
                      className={cn(
                        "group relative aspect-video rounded-xl overflow-hidden border-2 transition-all cursor-pointer",
                        activeSegmentIndex === i ? "border-emerald-500 shadow-lg shadow-emerald-500/20" : "border-zinc-800 hover:border-zinc-700"
                      )}
                    >
                      {seg.imageUrl ? (
                        <img 
                          src={seg.imageUrl} 
                          alt={`Segment ${i+1}`} 
                          className={cn(
                            "w-full h-full object-cover",
                            videoProvider === "CinematicPan" ? "animate-cinematic-pan" : ""
                          )} 
                          style={videoProvider === "CinematicPan" ? { 
                            animationDuration: `${(seg.duration || 5) * 2}s` 
                          } : {}}
                          referrerPolicy="no-referrer"
                        />
                      ) : (
                        <div className="w-full h-full bg-zinc-900 flex items-center justify-center">
                          {seg.isGeneratingImage ? <Loader2 className="w-4 h-4 text-emerald-500 animate-spin" /> : <ImageIcon className="w-4 h-4 text-zinc-800" />}
                        </div>
                      )}
                      <div className="absolute inset-x-0 bottom-0 p-2 bg-gradient-to-t from-black/80 to-transparent">
                        <span className="text-[8px] font-bold text-white uppercase tracking-widest">Parte {i + 1}</span>
                      </div>
                      {seg.videoUrl && (
                        <div className="absolute top-2 right-2 p-1 bg-emerald-500 rounded-md">
                          <Video className="w-2 h-2 text-black" />
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </main>

      {/* Footer */}
      <footer className="max-w-7xl mx-auto px-6 py-12 border-t border-zinc-800/50 mt-12">
        <div className="flex flex-col md:flex-row justify-between items-center gap-6">
          <div className="flex items-center gap-6">
            <div className="flex items-center gap-2 text-zinc-500 text-sm">
              <Sparkles className="w-4 h-4" />
              <span>StoryVisualizer AI v2.0</span>
            </div>
            <div className="h-4 w-px bg-zinc-800 hidden md:block" />
            <div 
              className="group relative flex items-center gap-2 px-3 py-1 bg-emerald-500/10 border border-emerald-500/20 rounded-full hover:bg-emerald-500/20 transition-all cursor-help"
            >
              <span className="text-[10px] font-bold text-emerald-500 uppercase tracking-wider">Costo Simulado:</span>
              <span className="text-sm font-mono font-bold text-emerald-400">{totalCost === 0 ? "GRATIS" : `$${totalCost.toFixed(3)}`}</span>
              
              {/* Tooltip Breakdown */}
              <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-56 bg-zinc-900 border border-zinc-800 rounded-xl p-3 shadow-2xl opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity z-50">
                <h4 className="text-[10px] font-bold uppercase text-zinc-500 mb-2 border-bottom border-zinc-800 pb-1">Uso de API (Free Tier)</h4>
                <div className="space-y-1.5">
                  <p className="text-[9px] text-zinc-400 mb-2 leading-relaxed">
                    Este costo es una estimación basada en precios estándar. Actualmente usas el nivel gratuito de Gemini.
                  </p>
                  <div className="flex justify-between text-[10px]">
                    <span className="text-zinc-400">Historias:</span>
                    <span className="text-white">{costBreakdown.stories}</span>
                  </div>
                  <div className="flex justify-between text-[10px]">
                    <span className="text-zinc-400">Imágenes:</span>
                    <span className="text-white">{costBreakdown.images}</span>
                  </div>
                  <div className="flex justify-between text-[10px]">
                    <span className="text-zinc-400">Videos:</span>
                    <span className="text-white">{costBreakdown.videos}</span>
                  </div>
                </div>
                <div className="mt-2 pt-2 border-t border-zinc-800 text-[9px] text-zinc-500 text-center">
                  Costo real: $0.00 (Nivel Gratuito)
                </div>
              </div>
              
              <a 
                href="https://console.cloud.google.com/billing" 
                target="_blank" 
                rel="noopener noreferrer"
                className="absolute inset-0"
                title="Ver facturación oficial en Google Cloud"
              />
            </div>
          </div>
          <div className="flex gap-8 text-xs font-bold uppercase tracking-widest text-zinc-600">
            <a href="#" className="hover:text-emerald-500 transition-colors">Privacy</a>
            <a href="#" className="hover:text-emerald-500 transition-colors">Terms</a>
            <a href="#" className="hover:text-emerald-500 transition-colors">Documentation</a>
          </div>
        </div>
      </footer>

      <ConfirmModal 
        isOpen={confirmModal.isOpen}
        title={confirmModal.title}
        message={confirmModal.message}
        onConfirm={confirmModal.onConfirm}
        onCancel={confirmModal.onCancel || (() => setConfirmModal(prev => ({ ...prev, isOpen: false })))}
        confirmText={confirmModal.confirmText}
        cancelText={confirmModal.cancelText}
        type={confirmModal.type}
      />

      <ExportModal 
        isOpen={isExportingVideo} 
        progress={exportProgress}
        currentScene={exportSceneInfo.current}
        totalScenes={exportSceneInfo.total}
        onCancel={() => {
          stopExportRef.current = true;
          setIsExportingVideo(false);
        }}
      />

      {story && (
        <PackagePreviewModal 
          isOpen={showPackagePreview}
          onClose={() => setShowPackagePreview(false)}
          onDownload={downloadAll}
          onExport={exportSingleVideo}
          story={story}
          totalCost={totalCost}
        />
      )}

      <AnimatePresence>
        {isPreviewingFinal && story && (
          <FinalPreviewPlayer 
            story={story}
            backgroundMusicMood={backgroundMusicMood}
            videoAspectRatio={videoAspectRatio}
            visualStyle={visualStyle}
            subtitleStyle={subtitleStyle}
            subtitlePosition={subtitlePosition}
            onDownload={downloadAll}
            onExport={exportSingleVideo}
            videoProvider={videoProvider}
            onClose={() => setIsPreviewingFinal(false)}
          />
        )}
      </AnimatePresence>
    </div>
  );
}

const ConfirmModal = ({ 
  isOpen, 
  title, 
  message, 
  onConfirm, 
  onCancel,
  confirmText = "Confirmar",
  cancelText = "Cancelar",
  type = 'info'
}: { 
  isOpen: boolean; 
  title: string; 
  message: string; 
  onConfirm: () => void; 
  onCancel: () => void;
  confirmText?: string;
  cancelText?: string;
  type?: 'danger' | 'info';
}) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm">
      <motion.div 
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className="w-full max-w-md bg-zinc-900 border border-zinc-800 rounded-3xl overflow-hidden shadow-2xl"
      >
        <div className="p-6 space-y-4">
          <div className="flex items-center gap-3">
            <div className={`w-10 h-10 rounded-full flex items-center justify-center ${type === 'danger' ? 'bg-red-500/10 text-red-500' : 'bg-emerald-500/10 text-emerald-500'}`}>
              {type === 'danger' ? <AlertTriangle className="w-5 h-5" /> : <AlertCircle className="w-5 h-5" />}
            </div>
            <h3 className="text-lg font-bold text-white">{title}</h3>
          </div>
          <p className="text-sm text-zinc-400 leading-relaxed">{message}</p>
        </div>
        <div className="p-4 bg-black/20 flex items-center justify-end gap-3">
          <button 
            onClick={onCancel}
            className="px-4 py-2 text-xs font-bold text-zinc-400 hover:text-white transition-colors"
          >
            {cancelText}
          </button>
          <button 
            onClick={onConfirm}
            className={`px-6 py-2 text-xs font-bold rounded-xl transition-all ${type === 'danger' ? 'bg-red-500 hover:bg-red-400 text-white' : 'bg-emerald-500 hover:bg-emerald-400 text-black'}`}
          >
            {confirmText}
          </button>
        </div>
      </motion.div>
    </div>
  );
};

const ExportModal = ({ 
  isOpen, 
  progress,
  onCancel,
  currentScene,
  totalScenes
}: { 
  isOpen: boolean; 
  progress: number;
  onCancel: () => void;
  currentScene?: number;
  totalScenes?: number;
}) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-[200] flex items-center justify-center p-4 bg-black/90 backdrop-blur-md">
      <motion.div 
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        className="w-full max-w-md bg-zinc-900 border border-zinc-800 rounded-3xl p-8 text-center space-y-6 shadow-2xl"
      >
        <div className="relative w-24 h-24 mx-auto">
          <svg className="w-full h-full transform -rotate-90">
            <circle
              cx="48"
              cy="48"
              r="40"
              stroke="currentColor"
              strokeWidth="8"
              fill="transparent"
              className="text-zinc-800"
            />
            <circle
              cx="48"
              cy="48"
              r="40"
              stroke="currentColor"
              strokeWidth="8"
              fill="transparent"
              strokeDasharray={251.2}
              strokeDashoffset={251.2 - (251.2 * progress) / 100}
              className="text-emerald-500 transition-all duration-300"
            />
          </svg>
          <div className="absolute inset-0 flex items-center justify-center">
            <span className="text-xl font-black text-white">{progress}%</span>
          </div>
        </div>

        <div className="space-y-2">
          <h3 className="text-xl font-black text-white uppercase tracking-tight">Exportando Video</h3>
          {currentScene !== undefined && totalScenes !== undefined && totalScenes > 0 && (
            <p className="text-emerald-400 font-mono text-sm">Procesando escena {currentScene} de {totalScenes}</p>
          )}
          <p className="text-sm text-zinc-400 leading-relaxed">
            Estamos uniendo todas las escenas, audios y subtítulos en un solo archivo de alta calidad.
          </p>
        </div>

        <div className="flex flex-col gap-3">
          <div className="flex items-center gap-3 p-4 bg-amber-500/10 border border-amber-500/20 rounded-2xl">
            <AlertTriangle className="w-5 h-5 text-amber-500 shrink-0" />
            <div className="text-left">
              <span className="text-[10px] font-bold text-amber-400 uppercase tracking-widest block">¡IMPORTANTE!</span>
              <span className="text-[9px] text-amber-200/70 leading-tight">Mantén esta pestaña activa y visible. Si cambias de pestaña, la exportación podría fallar o saltarse escenas.</span>
            </div>
          </div>

          <div className="flex items-center gap-3 p-4 bg-emerald-500/10 border border-emerald-500/20 rounded-2xl">
            <Loader2 className="w-5 h-5 text-emerald-500 animate-spin" />
            <div className="text-left">
              <span className="text-[10px] font-bold text-emerald-400 uppercase tracking-widest block">Procesando...</span>
              <span className="text-[9px] text-emerald-200/70 leading-tight">Estamos uniendo las piezas. Al finalizar, se descargará un archivo MP4 o WebM.</span>
            </div>
          </div>
          
          <div className="p-3 bg-white/5 border border-white/10 rounded-xl">
            <p className="text-[10px] text-white/50 leading-relaxed italic">
              Tip: Si el archivo final no abre en tu reproductor habitual, te recomendamos usar VLC Media Player o arrastrarlo a una pestaña de Chrome.
            </p>
          </div>
          
          <button 
            onClick={onCancel}
            className="px-6 py-3 bg-zinc-800 hover:bg-red-500/20 text-zinc-400 hover:text-red-500 text-xs font-bold rounded-xl transition-all border border-zinc-700/50 hover:border-red-500/50"
          >
            CANCELAR EXPORTACIÓN
          </button>
        </div>
      </motion.div>
    </div>
  );
};

const PackagePreviewModal = ({ 
  isOpen, 
  onClose, 
  onDownload,
  onExport,
  story,
  totalCost
}: { 
  isOpen: boolean; 
  onClose: () => void; 
  onDownload: () => void;
  onExport: () => void;
  story: Story;
  totalCost: number;
}) => {
  if (!isOpen) return null;

  const stats = {
    scenes: story.segments.length,
    images: story.segments.filter(s => s.imageUrl).length,
    videos: story.segments.filter(s => s.videoUrl).length,
    audio: !!story.audioUrl
  };

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm">
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="w-full max-w-lg bg-zinc-900 border border-zinc-800 rounded-3xl overflow-hidden shadow-2xl"
      >
        <div className="p-8 space-y-6">
          <div className="flex items-center justify-between">
            <div className="space-y-1">
              <h3 className="text-2xl font-black text-white uppercase tracking-tight">Resumen del Proyecto</h3>
              <p className="text-xs text-zinc-500 uppercase tracking-widest font-bold">{story.title}</p>
            </div>
            <button onClick={onClose} className="p-2 hover:bg-white/5 rounded-full transition-colors">
              <X className="w-5 h-5 text-zinc-500" />
            </button>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="p-4 bg-white/5 rounded-2xl border border-white/5 space-y-1">
              <span className="text-[10px] font-bold text-zinc-500 uppercase">Escenas Totales</span>
              <div className="text-xl font-black text-white">{stats.scenes}</div>
            </div>
            <div className="p-4 bg-white/5 rounded-2xl border border-white/5 space-y-1">
              <span className="text-[10px] font-bold text-zinc-500 uppercase">Inversión Estimada</span>
              <div className="text-xl font-black text-emerald-500">${totalCost.toFixed(2)}</div>
            </div>
          </div>

          <div className="space-y-3">
            <div className="flex items-center justify-between p-3 bg-black/20 rounded-xl border border-white/5">
              <div className="flex items-center gap-3">
                <ImageIcon className="w-4 h-4 text-zinc-400" />
                <span className="text-xs text-zinc-300">Imágenes Generadas</span>
              </div>
              <span className="text-xs font-bold text-white">{stats.images}/{stats.scenes}</span>
            </div>
            <div className="flex items-center justify-between p-3 bg-black/20 rounded-xl border border-white/5">
              <div className="flex items-center gap-3">
                <Video className="w-4 h-4 text-zinc-400" />
                <span className="text-xs text-zinc-300">Videos Animados</span>
              </div>
              <span className="text-xs font-bold text-white">{stats.videos}/{stats.scenes}</span>
            </div>
            <div className="flex items-center justify-between p-3 bg-black/20 rounded-xl border border-white/5">
              <div className="flex items-center gap-3">
                <Mic2 className="w-4 h-4 text-zinc-400" />
                <span className="text-xs text-zinc-300">Narración de Voz</span>
              </div>
              {stats.audio ? (
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="w-4 h-4 text-emerald-500" />
                  <button 
                    onClick={(e) => {
                      e.stopPropagation();
                      const audio = new Audio(story.audioUrl!);
                      audio.play();
                    }}
                    className="p-1.5 bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-500 rounded border border-emerald-500/20 transition-all"
                    title="Reproducir Narración"
                  >
                    <Play className="w-3.5 h-3.5" />
                  </button>
                  <button 
                    onClick={(e) => {
                      e.stopPropagation();
                      const link = document.createElement('a');
                      link.href = story.audioUrl!;
                      link.download = `${story.title.replace(/\s+/g, '_')}_narracion.wav`;
                      link.click();
                    }}
                    className="p-1.5 bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-500 rounded border border-emerald-500/20 transition-all"
                    title="Descargar Narración"
                  >
                    <Download className="w-3.5 h-3.5" />
                  </button>
                </div>
              ) : (
                <X className="w-4 h-4 text-red-500" />
              )}
            </div>
          </div>

          <div className="pt-4 flex flex-col gap-3">
            <button 
              onClick={() => {
                onExport();
                onClose();
              }}
              className="w-full py-4 bg-emerald-500 hover:bg-emerald-400 text-black font-black rounded-2xl flex items-center justify-center gap-3 transition-all shadow-xl shadow-emerald-500/20"
            >
              <Video className="w-5 h-5" />
              EXPORTAR VIDEO FINAL (MP4)
            </button>
            <button 
              onClick={() => {
                onDownload();
                onClose();
              }}
              className="w-full py-4 bg-white/5 hover:bg-white/10 text-white font-bold rounded-2xl flex items-center justify-center gap-3 transition-all border border-white/5"
            >
              <Download className="w-5 h-5" />
              Descargar Recursos (.ZIP)
            </button>
          </div>
        </div>
      </motion.div>
    </div>
  );
};
