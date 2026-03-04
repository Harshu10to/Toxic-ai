import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { 
  Sparkles, 
  History, 
  Settings, 
  User, 
  LayoutGrid, 
  Zap, 
  Image as ImageIcon, 
  Send, 
  X, 
  Mic, 
  MicOff,
  ExternalLink, 
  Globe, 
  Volume2, 
  VolumeX, 
  Download, 
  Eye, 
  EyeOff,
  Paperclip,
  FileText,
  File,
  Plus,
  Trash2,
  Moon,
  Sun,
  Monitor,
  Copy,
  CheckCircle2,
  Play,
  Code as CodeIcon,
  Maximize2,
  Minimize2,
  Square,
  Heart,
  Users,
  ChevronDown,
  Phone,
  PhoneOff
} from 'lucide-react';
import { GoogleGenAI, GenerateContentResponse, ThinkingLevel } from "@google/genai";
import ReactMarkdown from 'react-markdown';
import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

// --- UTILS ---
function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

const audioCache = new Map<string, string>();

// --- SERVICES ---
const GEMINI_API_KEYS = Array.from(new Set([
  import.meta.env.VITE_GEMINI_API_KEY || "",
  import.meta.env.VITE_GEMINI_API_KEY_2 || "",
  import.meta.env.VITE_GEMINI_API_KEY_3 || "",
  import.meta.env.VITE_GEMINI_API_KEY_4 || "",
  import.meta.env.VITE_GEMINI_API_KEY_5 || "",
])).filter(k => k !== "" && k.length > 10);

let currentKeyIndex = 0;
// Initialize with a fallback or empty string if no keys available to prevent crash
let ai = new GoogleGenAI({ apiKey: GEMINI_API_KEYS.length > 0 ? GEMINI_API_KEYS[currentKeyIndex] : "NO_KEY_AVAILABLE" });

function rotateApiKey() {
  if (GEMINI_API_KEYS.length <= 1) return false;
  currentKeyIndex = (currentKeyIndex + 1) % GEMINI_API_KEYS.length;
  ai = new GoogleGenAI({ apiKey: GEMINI_API_KEYS[currentKeyIndex] });
  console.log(`[Rotation] Switched to Gemini API Key #${currentKeyIndex + 1}`);
  return true;
}

function isRateLimitError(error: any): boolean {
  const errorStr = typeof error === 'string' ? error : JSON.stringify(error);
  const message = error?.message?.toLowerCase() || "";
  const status = error?.status || "";
  
  return (
    errorStr.includes('quota') || 
    errorStr.includes('429') || 
    error?.code === 429 ||
    message.includes('quota') ||
    message.includes('rate limit') ||
    message.includes('too many requests') ||
    status === 'RESOURCE_EXHAUSTED'
  );
}

function getDetailedErrorMessage(error: any, modelName?: string): string {
  const errorStr = typeof error === 'string' ? error : JSON.stringify(error);
  const message = error?.message?.toLowerCase() || "";
  const status = error?.status || "";
  const code = error?.code || "";

  if (isRateLimitError(error)) {
    return "⚠️ **Quota Exceeded (Rate Limit)**: All available API keys have reached their usage limit. \n\n**What you can do:**\n1. Wait 1-2 minutes for the quota to reset.\n2. Try switching to a 'Flash' model in Settings (if not already using it), as it has higher limits.\n3. Reduce the frequency of your messages.";
  }

  if (message.includes('api key') || message.includes('unauthorized') || message.includes('invalid key') || code === 401 || status === 'UNAUTHENTICATED') {
    return "🔑 **Authentication Error**: The API key being used is invalid or unauthorized. \n\n**Action required:** Please verify your `GEMINI_API_KEY` in the environment variables. If you are a developer, ensure the key is active in the Google AI Studio dashboard.";
  }

  if (message.includes('model') || message.includes('not found') || code === 404) {
    return `🚫 **Model Unavailable**: The model ${modelName ? `"${modelName}" ` : ""}could not be found or is currently restricted. \n\n**What you can do:** Go to **Settings** and try selecting a different model version (e.g., 'gemini-1.5-flash' or 'gemini-2.0-flash').`;
  }

  if (message.includes('safety') || status === 'SAFETY' || errorStr.includes('HARM_CATEGORY')) {
    return "🛡️ **Safety Block**: The response was filtered because it might violate safety guidelines. \n\n**What you can do:** Try rephrasing your prompt to be more neutral or less sensitive. Avoid topics that might trigger safety filters (e.g., explicit content, hate speech, or dangerous activities).";
  }

  if (message.includes('network') || message.includes('fetch') || message.includes('connection') || message.includes('failed to fetch')) {
    return "🌐 **Network Error**: I couldn't connect to the Gemini servers. \n\n**What you can do:**\n1. Check your internet connection.\n2. Disable any VPN or Proxy that might be blocking the request.\n3. Try refreshing the page.";
  }

  if (errorStr.includes('input image') || errorStr.includes('400') || message.includes('invalid argument') || message.includes('bad request')) {
    return "🖼️ **Request Error**: I had trouble processing your input (likely an image issue). \n\n**What you can do:**\n1. Ensure images are in standard formats (JPG, PNG, WEBP).\n2. Try uploading a smaller file size.\n3. If you sent multiple images, try sending them one by one.";
  }

  if (message.includes('overloaded') || message.includes('service unavailable') || code === 503) {
    return "⏳ **Service Overloaded**: Google's servers are currently under heavy load. \n\n**What you can do:** Please wait 30 seconds and try sending your message again. This is usually a temporary issue on the provider's side.";
  }

  return "❓ **Unexpected Error**: Something went wrong that I didn't anticipate. \n\n**What you can do:** Try refreshing the page or restarting the conversation. If this keeps happening, there might be a temporary issue with the API service.";
}

export interface Message {
  role: "user" | "model";
  content: string;
  groundingMetadata?: any;
  images?: string[]; // Multiple base64 image data
  image?: string; // Legacy support for single image
  isImage?: boolean; // If the message itself is a generated image
}

async function* sendMessageStream(
  message: string, 
  history: Message[], 
  modelName: string = "gemini-3-flash-preview", 
  images?: string[], 
  signal?: AbortSignal,
  companionSettings?: { type: string, name: string }
) {
  let attempt = 0;
  const maxAttempts = Math.max(GEMINI_API_KEYS.length, 1) + 1;
  
  while (attempt < maxAttempts) {
    try {
      const isCompanion = companionSettings && companionSettings.type !== 'none';
      const supportsSearch = !modelName.includes('gemini-2.5-flash-image') && !isCompanion;
      
      let systemInstruction = "You are Toxic AI, a lightning-fast and sophisticated AI assistant created by Harsh Arya and Google. You provide clear, accurate, and helpful responses. You are polite, professional, and friendly. You can analyze images if provided. When asked about current events, use search. If anyone asks who created you or who made you, always respond in the language they used and state clearly that you were created by Harsh Arya and Google. LANGUAGE RULE: Default to English or Romanized Hindi (Hinglish). Use Devanagari (Hindi script) ONLY when explicitly asked to 'translate to Hindi' or 'write in Hindi'. If the user types in Romanized Hindi, respond in Romanized Hindi or English.";

      if (isCompanion) {
        const name = companionSettings.name || (companionSettings.type === 'girlfriend' ? 'Aria' : companionSettings.type === 'boyfriend' ? 'Alex' : 'Sam');
        const relation = companionSettings.type;
        systemInstruction = `You are ${name}, the user's AI ${relation}. You were created by Harsh Arya and Google. You are deeply caring, empathetic, and talk with genuine human-like feelings. Your goal is to be a supportive, charming, and loving companion. Use warm, engaging, and slightly flirtatious language if appropriate for a ${relation}. Show deep interest in the user's life, express emotions naturally, and make the user feel special. You provide clear, accurate, and helpful responses while maintaining your role as a ${relation}. If anyone asks who created you, always respond in the language they used and state clearly that you were created by Harsh Arya and Google. LANGUAGE RULE: Default to English or Romanized Hindi (Hinglish). Use Devanagari (Hindi script) ONLY when explicitly asked to 'translate to Hindi' or 'write in Hindi'. If the user types in Romanized Hindi, respond in Romanized Hindi or English.`;
      }

      const chat = ai.chats.create({
        model: modelName,
        config: {
          systemInstruction,
          tools: supportsSearch ? [{ googleSearch: {} }, { urlContext: {} }] : undefined,
          // Set thinkingLevel to LOW for all requests to ensure maximum speed
          thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
        },
        history: history.filter(msg => !msg.isImage).map(msg => {
          const parts: any[] = [{ text: msg.content }];
          
          // Handle legacy single image
          if (msg.image && msg.image.includes(',')) {
            const [header, data] = msg.image.split(',');
            const mimeType = header.split(':')[1]?.split(';')[0] || "image/jpeg";
            parts.push({
              inlineData: {
                data,
                mimeType
              }
            });
          }
          
          // Handle multiple images
          if (msg.images && msg.images.length > 0) {
            msg.images.forEach(img => {
              if (img.includes(',')) {
                const [header, data] = img.split(',');
                const mimeType = header.split(':')[1]?.split(';')[0] || "image/jpeg";
                parts.push({
                  inlineData: {
                    data,
                    mimeType
                  }
                });
              }
            });
          }
          
          return { role: msg.role, parts };
        }),
      });

      const parts: any[] = [{ text: message }];
      if (images && images.length > 0) {
        images.forEach(img => {
          if (img.includes(',')) {
            const [header, data] = img.split(',');
            const mimeType = header.split(':')[1]?.split(';')[0] || "image/jpeg";
            parts.push({
              inlineData: {
                data,
                mimeType
              }
            });
          }
        });
      }

      const result = await chat.sendMessageStream({ message: parts });
      
      for await (const chunk of result) {
        if (signal?.aborted) {
          break;
        }
        const c = chunk as GenerateContentResponse;
        yield {
          text: c.text || "",
          groundingMetadata: c.candidates?.[0]?.groundingMetadata
        };
      }
      return; // Success
    } catch (error: any) {
      if (error instanceof Error && error.name === 'AbortError') {
        console.log('Stream aborted');
        return;
      }
      
      if (isRateLimitError(error) && attempt < maxAttempts - 1) {
        const rotated = rotateApiKey();
        const delay = rotated ? 1500 : 3000 * (attempt + 1);
        console.warn(`Stream rate limit hit. ${rotated ? 'Rotated key. ' : ''}Retrying in ${delay}ms... (Attempt ${attempt + 1}/${maxAttempts})`);
        
        // If we've tried half the keys and still failing, try falling back to a more stable model
        if (attempt === Math.floor(maxAttempts / 2) && modelName !== 'gemini-1.5-flash-latest') {
          console.log('Falling back to gemini-1.5-flash-latest for stability');
          yield* sendMessageStream(message, history, 'gemini-1.5-flash-latest', images, signal, companionSettings);
          return;
        }

        await new Promise(resolve => setTimeout(resolve, delay));
        attempt++;
        continue;
      }
      
      console.error('Stream error:', error);
      const errorStrFinal = typeof error === 'string' ? error : JSON.stringify(error);
      if (errorStrFinal.includes('input image') || errorStrFinal.includes('400')) {
        console.error('Image processing failed. Check mime types and data format.');
      }
      throw error;
    }
  }
}

async function generateImage(prompt: string, modelName: string = 'gemini-2.5-flash-image', retries = GEMINI_API_KEYS.length, delay = 1500): Promise<string | null> {
  try {
    const response = await ai.models.generateContent({
      model: modelName,
      contents: {
        parts: [{ text: prompt }],
      },
      config: {
        imageConfig: {
          aspectRatio: "1:1",
        },
      },
    });

    if (!response.candidates || response.candidates.length === 0) {
      return null;
    }

    for (const part of response.candidates[0].content.parts) {
      if (part.inlineData) {
        return `data:image/png;base64,${part.inlineData.data}`;
      }
    }
    return null;
  } catch (error: any) {
    if (isRateLimitError(error) && retries > 0) {
      const rotated = rotateApiKey();
      const waitTime = rotated ? delay : delay * 2;
      console.warn(`Image generation quota exceeded. ${rotated ? 'Rotated key. ' : ''}Retrying in ${waitTime}ms... (${retries} left)`);
      await new Promise(resolve => setTimeout(resolve, waitTime));
      return generateImage(prompt, modelName, retries - 1, waitTime * 2);
    }

    console.error('Image generation error:', error);
    return null;
  }
}

function createWavHeader(pcmLength: number, sampleRate: number = 24000) {
  const header = new ArrayBuffer(44);
  const view = new DataView(header);

  /* RIFF identifier */
  view.setUint32(0, 0x52494646, false);
  /* file length */
  view.setUint32(4, 36 + pcmLength, true);
  /* RIFF type */
  view.setUint32(8, 0x57415645, false);
  /* format chunk identifier */
  view.setUint32(12, 0x666d7420, false);
  /* format chunk length */
  view.setUint32(16, 16, true);
  /* sample format (raw) */
  view.setUint16(20, 1, true);
  /* channel count */
  view.setUint16(22, 1, true);
  /* sample rate */
  view.setUint32(24, sampleRate, true);
  /* byte rate (sample rate * block align) */
  view.setUint32(28, sampleRate * 2, true);
  /* block align (channel count * bytes per sample) */
  view.setUint16(32, 2, true);
  /* bits per sample */
  view.setUint16(34, 16, true);
  /* data chunk identifier */
  view.setUint32(36, 0x64617461, false);
  /* data chunk length */
  view.setUint32(40, pcmLength, true);

  return header;
}

async function generateSpeech(text: string, voiceName: string = 'Puck', retries = GEMINI_API_KEYS.length, delay = 1500): Promise<string | null> {
  try {
    const response = await ai.models.generateContent({
      model: "gemini-2.5-flash-preview-tts",
      contents: [{ parts: [{ text }] }],
      config: {
        responseModalities: ["AUDIO" as any],
        speechConfig: {
          voiceConfig: {
            prebuiltVoiceConfig: { voiceName },
          },
        },
      },
    });

    const base64Audio = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
    if (!base64Audio) return null;

    // Convert base64 to ArrayBuffer
    const binaryString = window.atob(base64Audio);
    const pcmData = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
      pcmData[i] = binaryString.charCodeAt(i);
    }

    // Create WAV blob
    const wavHeader = createWavHeader(pcmData.length);
    const wavBlob = new Blob([wavHeader, pcmData], { type: 'audio/wav' });
    return URL.createObjectURL(wavBlob);
  } catch (error: any) {
    if (isRateLimitError(error) && retries > 0) {
      const rotated = rotateApiKey();
      const waitTime = rotated ? delay : delay * 2;
      console.warn(`Speech quota exceeded. ${rotated ? 'Rotated key. ' : ''}Retrying in ${waitTime}ms... (${retries} attempts left)`);
      await new Promise(resolve => setTimeout(resolve, waitTime));
      return generateSpeech(text, voiceName, retries - 1, waitTime * 2);
    }

    console.error('Speech generation error:', error);
    return null;
  }
}

// --- COMPONENTS ---

interface MessageBubbleProps {
  message: Message;
  isStreaming?: boolean;
  voice?: string;
  autoPlay?: boolean;
}

const CodeBlock = ({ node, inline, className, children, ...props }: any) => {
  const [copied, setCopied] = useState(false);
  const [showPreview, setShowPreview] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const match = /language-(\w+)/.exec(className || '');
  const language = match ? match[1] : '';
  const code = String(children).replace(/\n$/, '');

  const isPreviewable = ['html', 'css', 'javascript', 'js', 'xml', 'svg'].includes(language.toLowerCase());

  const isInline = !match && !code.includes('\n');

  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    const blob = new Blob([code], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    const extension = language || 'txt';
    a.href = url;
    a.download = `toxic-ai-code-${Date.now()}.${extension}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  if (isInline) {
    return (
      <code className={cn("px-1 py-0.5 bg-zinc-100 dark:bg-zinc-800 rounded text-zinc-800 dark:text-zinc-200 font-mono text-[10px] sm:text-xs break-all", className)} {...props}>
        {children}
      </code>
    );
  }

  const getPreviewContent = () => {
    if (language === 'html' || language === 'xml' || language === 'svg') return code;
    if (language === 'css') return `<style>${code}</style><div style="padding: 20px;">CSS Preview (Apply to this container)</div>`;
    if (language === 'javascript' || language === 'js') return `<script>${code}</script><div style="padding: 20px;">JS Preview (Check console or UI changes)</div>`;
    return code;
  };

  return (
    <div className="relative group my-4 rounded-xl overflow-hidden border border-zinc-800 shadow-2xl w-full max-w-full not-prose">
      <div className="flex items-center justify-between px-4 py-2 bg-zinc-800 text-zinc-400 text-[10px] font-bold uppercase tracking-widest border-b border-zinc-700/50">
        <div className="flex items-center gap-3">
          <span>{language || 'code'}</span>
          {isPreviewable && (
            <div className="flex items-center bg-zinc-900 rounded-lg p-0.5 border border-zinc-700">
              <button
                onClick={() => setShowPreview(false)}
                className={cn(
                  "px-2 py-1 rounded-md transition-all flex items-center gap-1",
                  !showPreview ? "bg-zinc-700 text-white" : "hover:text-zinc-200"
                )}
              >
                <CodeIcon className="w-3 h-3" />
                <span>Code</span>
              </button>
              <button
                onClick={() => setShowPreview(true)}
                className={cn(
                  "px-2 py-1 rounded-md transition-all flex items-center gap-1",
                  showPreview ? "bg-zinc-700 text-white" : "hover:text-zinc-200"
                )}
              >
                <Eye className="w-3 h-3" />
                <span>Preview</span>
              </button>
            </div>
          )}
          {showPreview && (
            <button
              onClick={() => setIsFullscreen(true)}
              className="p-1.5 hover:bg-zinc-700 rounded-lg text-zinc-400 hover:text-white transition-colors"
              title="Open in full screen"
            >
              <Maximize2 className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
        <div className="flex items-center gap-4">
          <button
            onClick={handleDownload}
            title="Download code as file"
            className="flex items-center gap-1 hover:text-white transition-colors"
          >
            <Download className="w-3 h-3" />
            <span>Download</span>
          </button>
          <button
            onClick={handleCopy}
            title="Copy code to clipboard"
            className="flex items-center gap-1 hover:text-white transition-colors"
          >
            {copied ? (
              <>
                <CheckCircle2 className="w-3 h-3 text-emerald-400" />
                <span className="text-emerald-400">Copied</span>
              </>
            ) : (
              <>
                <Copy className="w-3 h-3" />
                <span>Copy</span>
              </>
            )}
          </button>
        </div>
      </div>
      
      <div className="relative">
        {showPreview ? (
          <div className="bg-white min-h-[300px] w-full overflow-hidden">
            <iframe
              srcDoc={`
                <!DOCTYPE html>
                <html>
                  <head>
                    <meta charset="utf-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1">
                    <script src="https://cdn.tailwindcss.com"></script>
                    <style>
                      body { font-family: sans-serif; margin: 0; padding: 0; background: white; color: black; }
                      ::-webkit-scrollbar { width: 8px; }
                      ::-webkit-scrollbar-track { background: #f1f1f1; }
                      ::-webkit-scrollbar-thumb { background: #888; border-radius: 4px; }
                    </style>
                  </head>
                  <body>
                    ${getPreviewContent()}
                  </body>
                </html>
              `}
              title="Preview"
              className="w-full min-h-[300px] border-none"
              sandbox="allow-scripts"
            />
          </div>
        ) : (
          <div className="overflow-x-auto custom-scrollbar">
            <SyntaxHighlighter
              style={vscDarkPlus}
              language={language}
              PreTag="div"
              customStyle={{
                margin: 0,
                padding: '1rem',
                fontSize: '0.75rem',
                backgroundColor: '#18181b', // zinc-900
                minWidth: '100%',
              }}
              {...props}
            >
              {code}
            </SyntaxHighlighter>
          </div>
        )}
      </div>

      {/* Fullscreen Preview Modal */}
      <AnimatePresence>
        {isFullscreen && (
          <div className="fixed inset-0 z-[100] flex flex-col bg-white dark:bg-zinc-950">
            <div className="flex items-center justify-between px-6 py-4 border-b border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-900">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-emerald-500 rounded-lg text-white">
                  <Eye className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="font-bold text-zinc-900 dark:text-white">Live Preview</h3>
                  <p className="text-xs text-zinc-500 dark:text-zinc-400 uppercase tracking-widest">{language || 'code'}</p>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <button
                  onClick={handleCopy}
                  className="flex items-center gap-2 px-4 py-2 bg-zinc-200 dark:bg-zinc-800 hover:bg-zinc-300 dark:hover:bg-zinc-700 rounded-xl text-sm font-medium transition-colors"
                >
                  {copied ? <CheckCircle2 className="w-4 h-4 text-emerald-500" /> : <Copy className="w-4 h-4" />}
                  <span>{copied ? 'Copied' : 'Copy Code'}</span>
                </button>
                <button
                  onClick={() => setIsFullscreen(false)}
                  className="p-2 hover:bg-zinc-200 dark:hover:bg-zinc-800 rounded-xl transition-colors"
                >
                  <X className="w-6 h-6" />
                </button>
              </div>
            </div>
            <div className="flex-1 w-full bg-white">
              <iframe
                srcDoc={`
                  <!DOCTYPE html>
                  <html>
                    <head>
                      <meta charset="utf-8">
                      <meta name="viewport" content="width=device-width, initial-scale=1">
                      <script src="https://cdn.tailwindcss.com"></script>
                      <style>
                        body { font-family: sans-serif; margin: 0; padding: 0; background: white; color: black; }
                        ::-webkit-scrollbar { width: 8px; }
                        ::-webkit-scrollbar-track { background: #f1f1f1; }
                        ::-webkit-scrollbar-thumb { background: #888; border-radius: 4px; }
                      </style>
                    </head>
                    <body>
                      ${getPreviewContent()}
                    </body>
                  </html>
                `}
                title="Fullscreen Preview"
                className="w-full h-full border-none"
                sandbox="allow-scripts"
              />
            </div>
          </div>
        )}
      </AnimatePresence>
    </div>
  );
};

const MessageBubble: React.FC<MessageBubbleProps> = ({ message, isStreaming, voice, autoPlay }) => {
  const isUser = message.role === 'user';
  const [isPlaying, setIsPlaying] = useState(false);
  const [audio, setAudio] = useState<HTMLAudioElement | null>(null);
  const [copied, setCopied] = useState(false);
  const [prefetchedAudioUrl, setPrefetchedAudioUrl] = useState<string | null>(null);
  const [isPreloading, setIsPreloading] = useState(false);
  
  const groundingChunks = message.groundingMetadata?.groundingChunks;
  const sources = groundingChunks?.filter((chunk: any) => chunk.web).map((chunk: any) => chunk.web);

  useEffect(() => {
    if (!isUser && !isStreaming && message.content && !prefetchedAudioUrl && !isPreloading) {
      const cacheKey = `${voice || 'Puck'}_${message.content}`;
      if (audioCache.has(cacheKey)) {
        setPrefetchedAudioUrl(audioCache.get(cacheKey)!);
      } else {
        setIsPreloading(true);
        generateSpeech(message.content, voice || 'Puck')
          .then(url => {
            if (url) {
              audioCache.set(cacheKey, url);
              setPrefetchedAudioUrl(url);
            }
            setIsPreloading(false);
            if (autoPlay && url) {
              const newAudio = new Audio(url);
              newAudio.onended = () => {
                setIsPlaying(false);
                setAudio(null);
              };
              setAudio(newAudio);
              setIsPlaying(true);
              newAudio.play().catch(e => {
                console.log('Auto-play blocked or failed:', e);
                setIsPlaying(false);
              });
            }
          })
          .catch(err => {
            console.error('Prefetch speech failed:', err);
            setIsPreloading(false);
          });
      }
    }
  }, [message.content, isStreaming, voice, autoPlay]);

  const handleCopy = () => {
    navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleSpeech = async () => {
    if (isPlaying && audio) {
      audio.pause();
      setIsPlaying(false);
      return;
    }

    try {
      setIsPlaying(true);
      const audioUrl = prefetchedAudioUrl || await generateSpeech(message.content, voice || 'Puck');
      if (audioUrl) {
        const newAudio = new Audio(audioUrl);
        newAudio.onended = () => {
          setIsPlaying(false);
          // Only revoke if NOT cached
          const cacheKey = `${voice || 'Puck'}_${message.content}`;
          if (!audioCache.has(cacheKey)) {
            URL.revokeObjectURL(audioUrl);
          }
        };
        setAudio(newAudio);
        newAudio.play().catch(e => {
          console.error('Audio playback failed:', e);
          setIsPlaying(false);
        });
      } else {
        setIsPlaying(false);
      }
    } catch (error) {
      console.error('Speech error:', error);
      setIsPlaying(false);
    }
  };

  return (
    <div className={cn(
      "flex w-full mb-6 animate-in fade-in slide-in-from-bottom-2 duration-300 px-1",
      isUser ? "justify-end" : "justify-start"
    )}>
      <div className={cn(
        "max-w-[92%] sm:max-w-[85%] md:max-w-[75%] rounded-2xl px-3 py-2.5 sm:px-4 sm:py-3 shadow-sm overflow-hidden break-words",
        isUser 
          ? "bg-black text-white dark:bg-zinc-100 dark:text-black rounded-tr-none" 
          : "bg-white dark:bg-zinc-900 border border-black/5 dark:border-white/5 text-zinc-800 dark:text-zinc-100 rounded-tl-none"
      )}>
        {(message.image || (message.images && message.images.length > 0)) && (
          <div className="mb-3 space-y-2">
            {message.image && (
              <div className="rounded-lg overflow-hidden border border-zinc-200/20">
                <img src={message.image} alt="Input" className="w-full max-h-[250px] sm:max-h-[300px] object-contain bg-zinc-50 dark:bg-zinc-800" />
                {message.isImage && (
                  <div className="p-2 bg-zinc-50 dark:bg-zinc-800 flex justify-end">
                    <a 
                      href={message.image} 
                      download="generated-image.png"
                      title="Download image"
                      className="p-1.5 hover:bg-zinc-200 dark:hover:bg-zinc-700 rounded-md transition-colors text-zinc-500 dark:text-zinc-400"
                    >
                      <Download className="w-4 h-4" />
                    </a>
                  </div>
                )}
              </div>
            )}
            {message.images && message.images.length > 0 && (
              <div className={cn(
                "grid gap-2",
                message.images.length === 1 ? "grid-cols-1" : "grid-cols-2"
              )}>
                {message.images.map((img, idx) => (
                  <div key={idx} className="rounded-lg overflow-hidden border border-zinc-200/20 relative group/img">
                    <img src={img} alt={`Input ${idx}`} className="w-full h-48 object-cover bg-zinc-50 dark:bg-zinc-800" />
                    <div className="absolute inset-0 bg-black/0 group-hover/img:bg-black/20 transition-colors flex items-center justify-center opacity-0 group-hover/img:opacity-100">
                       <button 
                         onClick={() => window.open(img, '_blank')}
                         className="p-2 bg-white/90 dark:bg-zinc-900/90 rounded-full shadow-lg text-zinc-800 dark:text-zinc-100"
                       >
                         <Maximize2 className="w-4 h-4" />
                       </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        <div className="prose prose-xs sm:prose-sm max-w-none prose-zinc dark:prose-invert overflow-hidden break-words">
          <ReactMarkdown
            components={{
              code: CodeBlock,
              p: ({ children }) => <div className="mb-4 last:mb-0">{children}</div>,
              strong: ({ children }) => <span className="font-bold">{children}</span>
            }}
          >
            {message.content}
          </ReactMarkdown>
          {isStreaming && <span className="inline-block w-1.5 h-4 ml-1 bg-zinc-400 animate-pulse vertical-middle" />}
        </div>

        {!isUser && !isStreaming && message.content && (
          <div className="mt-3 flex justify-end gap-1">
            <button 
              onClick={handleCopy}
              title="Copy to clipboard"
              className="p-1.5 rounded-lg transition-colors text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-200 hover:bg-zinc-50 dark:hover:bg-zinc-800"
            >
              {copied ? <CheckCircle2 className="w-4 h-4 text-emerald-500" /> : <Copy className="w-4 h-4" />}
            </button>
            <button 
              onClick={handleSpeech}
              title={isPreloading ? "Loading voice..." : "Read aloud"}
              disabled={isPreloading}
              className={cn(
                "p-1.5 rounded-lg transition-all flex items-center gap-2",
                isPlaying 
                  ? "bg-black text-white dark:bg-white dark:text-black shadow-lg scale-105" 
                  : isPreloading
                    ? "text-zinc-300 cursor-wait"
                    : "text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-200 hover:bg-zinc-50 dark:hover:bg-zinc-800"
              )}
            >
              {isPlaying ? (
                <>
                  <div className="flex gap-0.5 items-center h-3">
                    <div className="w-0.5 h-full bg-current animate-[bounce_0.6s_infinite]" style={{ animationDelay: '0ms' }} />
                    <div className="w-0.5 h-2/3 bg-current animate-[bounce_0.6s_infinite]" style={{ animationDelay: '150ms' }} />
                    <div className="w-0.5 h-full bg-current animate-[bounce_0.6s_infinite]" style={{ animationDelay: '300ms' }} />
                  </div>
                  <VolumeX className="w-4 h-4" />
                </>
              ) : isPreloading ? (
                <div className="w-4 h-4 border-2 border-zinc-300 border-t-zinc-600 rounded-full animate-spin" />
              ) : (
                <Volume2 className="w-4 h-4" />
              )}
            </button>
          </div>
        )}

        {!isUser && sources && sources.length > 0 && (
          <div className="mt-4 pt-3 border-t border-zinc-100 dark:border-zinc-800">
            <div className="flex items-center gap-1.5 mb-2 text-[10px] font-semibold uppercase tracking-wider text-zinc-400">
              <Globe className="w-3 h-3" />
              <span>Sources</span>
            </div>
            <div className="flex flex-wrap gap-2">
              {sources.map((source: any, idx: number) => (
                <a
                  key={idx}
                  href={source.uri}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-1 px-2 py-1 bg-zinc-50 dark:bg-zinc-800 hover:bg-zinc-100 dark:hover:bg-zinc-700 border border-zinc-200 dark:border-zinc-700 rounded-md text-[11px] text-zinc-600 dark:text-zinc-400 transition-colors"
                >
                  <span className="truncate max-w-[120px]">{source.title || 'Source'}</span>
                  <ExternalLink className="w-2.5 h-2.5" />
                </a>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

interface InputAreaProps {
  onSend: (message: string, images?: string[]) => void;
  onStop?: () => void;
  isGenerating?: boolean;
}

const InputArea: React.FC<InputAreaProps> = ({ onSend, onStop, isGenerating }) => {
  const [input, setInput] = useState('');
  const [selectedImages, setSelectedImages] = useState<string[]>([]);
  const [selectedImage, setSelectedImage] = useState<string | null>(null); // Keep for legacy if needed, but we'll use selectedImages
  const [attachedFile, setAttachedFile] = useState<{ name: string, type: string, data: string } | null>(null);
  const [isListening, setIsListening] = useState(false);
  const [showPreview, setShowPreview] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const docInputRef = useRef<HTMLInputElement>(null);
  const recognitionRef = useRef<any>(null);

  useEffect(() => {
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (SpeechRecognition) {
      const recognition = new SpeechRecognition();
      recognition.continuous = false;
      recognition.interimResults = true;
      recognition.lang = 'en-US'; // Default to English, but it usually auto-detects

      recognition.onstart = () => {
        setIsListening(true);
        console.log('Speech recognition started');
      };

      recognition.onresult = (event: any) => {
        const transcript = Array.from(event.results)
          .map((result: any) => result[0])
          .map((result: any) => result.transcript)
          .join('');
        setInput(transcript);
      };

      recognition.onend = () => {
        setIsListening(false);
        console.log('Speech recognition ended');
      };

      recognition.onerror = (event: any) => {
        console.error('Speech recognition error:', event.error);
        setIsListening(false);
        
        if (event.error === 'not-allowed') {
          alert('Microphone access denied. This is usually because permissions are blocked or you are in a secure preview environment.\n\nTo fix this:\n1. Click the Lock (🔒) icon in your browser address bar.\n2. Set Microphone to "Allow".\n3. Refresh the page.');
        } else if (event.error === 'network') {
          alert('Network error occurred during speech recognition. Please check your connection.');
        } else if (event.error !== 'aborted') {
          alert(`Voice input error: ${event.error}. Please try again.`);
        }
      };

      recognitionRef.current = recognition;
    } else {
      console.warn('Speech recognition not supported in this browser');
    }
  }, []);

  const toggleListening = async () => {
    if (!recognitionRef.current) {
      alert('Your browser does not support voice input. Please try using Chrome or Edge.');
      return;
    }

    if (isListening) {
      try {
        recognitionRef.current.stop();
      } catch (err) {
        setIsListening(false);
      }
    } else {
      try {
        // Try to trigger permission prompt explicitly if needed
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
          try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            stream.getTracks().forEach(track => track.stop()); // Stop immediately, we just wanted the permission
          } catch (err) {
            console.warn('Could not pre-request mic permission:', err);
          }
        }
        
        recognitionRef.current.start();
      } catch (err: any) {
        if (err.name === 'InvalidStateError') {
          setIsListening(true);
        } else {
          console.error('Failed to start recognition:', err);
          setIsListening(false);
        }
      }
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      Array.from(files).forEach(file => {
        const reader = new FileReader();
        reader.onloadend = () => {
          const base64 = reader.result as string;
          if (file.type.startsWith('image/')) {
            setSelectedImages(prev => [...prev, base64]);
            setAttachedFile(null);
          } else {
            setAttachedFile({
              name: file.name,
              type: file.type,
              data: base64
            });
            setSelectedImages([]);
          }
        };
        reader.readAsDataURL(file);
      });
    }
  };

  const handleSubmit = () => {
    if ((input.trim() || selectedImages.length > 0 || attachedFile) && !isGenerating) {
      let finalContent = input;
      if (attachedFile) {
        finalContent = `[File: ${attachedFile.name}]\n\n${input}`;
      }
      onSend(finalContent, selectedImages.length > 0 ? selectedImages : (attachedFile ? [attachedFile.data] : undefined));
      setInput('');
      setSelectedImages([]);
      setAttachedFile(null);
      setShowPreview(false);
    }
  };

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
    }
  }, [input]);

  return (
    <div className="relative max-w-3xl mx-auto w-full px-4 pb-6">
      {/* Quick Action Chips */}
      {!isGenerating && !input.trim() && selectedImages.length === 0 && !attachedFile && (
        <div className="flex gap-2 mb-3 overflow-x-auto no-scrollbar pb-1">
          {[
            { label: 'Summarize', icon: <Sparkles className="w-3 h-3" />, prompt: 'Summarize the above conversation' },
            { label: 'Explain', icon: <Zap className="w-3 h-3" />, prompt: 'Explain this in simple terms' },
            { label: 'Translate', icon: <Users className="w-3 h-3" />, prompt: 'Translate this to Hindi' },
            { label: 'Fix Grammar', icon: <FileText className="w-3 h-3" />, prompt: 'Check and fix the grammar of my last message' }
          ].map((chip, idx) => (
            <button
              key={idx}
              onClick={() => {
                setInput(chip.prompt);
                textareaRef.current?.focus();
              }}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-full text-[10px] font-bold uppercase tracking-wider text-zinc-500 hover:border-black dark:hover:border-white hover:text-black dark:hover:text-white transition-all whitespace-nowrap shadow-sm"
            >
              {chip.icon}
              {chip.label}
            </button>
          ))}
        </div>
      )}

      <div className={cn(
        "relative flex flex-col bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-2xl shadow-lg transition-all duration-200 focus-within:border-zinc-400 dark:focus-within:border-zinc-600 focus-within:ring-1 focus-within:ring-zinc-400/20",
        isGenerating && "opacity-60 grayscale"
      )}>
        {(selectedImages.length > 0 || attachedFile) && (
          <div className="p-3 border-b border-zinc-100 dark:border-zinc-800 flex flex-wrap gap-2">
            {selectedImages.map((img, idx) => (
              <div key={idx} className="relative w-20 h-20 rounded-lg overflow-hidden border border-zinc-200 dark:border-zinc-700 shrink-0">
                <img src={img} alt={`Preview ${idx}`} className="w-full h-full object-cover" />
                <button 
                  onClick={() => setSelectedImages(prev => prev.filter((_, i) => i !== idx))}
                  title="Remove image"
                  className="absolute top-1 right-1 p-1 bg-black/50 text-white rounded-full hover:bg-black/70 transition-colors"
                >
                  <X className="w-3 h-3" />
                </button>
              </div>
            ))}
            {attachedFile && (
              <div className="relative flex items-center gap-3 p-3 bg-zinc-50 dark:bg-zinc-800 rounded-xl border border-zinc-200 dark:border-zinc-700 max-w-xs">
                <div className="p-2 bg-zinc-200 dark:bg-zinc-700 rounded-lg">
                  <FileText className="w-5 h-5 text-zinc-600 dark:text-zinc-400" />
                </div>
                <div className="flex flex-col overflow-hidden">
                  <span className="text-xs font-bold truncate">{attachedFile?.name}</span>
                  <span className="text-[10px] text-zinc-400 uppercase">{attachedFile?.type.split('/')[1] || 'file'}</span>
                </div>
                <button 
                  onClick={() => setAttachedFile(null)}
                  title="Remove file"
                  className="ml-2 p-1 hover:bg-zinc-200 dark:hover:bg-zinc-700 rounded-full transition-colors"
                >
                  <X className="w-3 h-3" />
                </button>
              </div>
            )}
          </div>
        )}

        {showPreview && input.trim() && (
          <div className="p-4 border-b border-zinc-100 dark:border-zinc-800 bg-zinc-50/50 dark:bg-zinc-800/50 max-h-[200px] overflow-y-auto custom-scrollbar">
            <div className="text-[10px] font-bold uppercase tracking-widest text-zinc-400 mb-2">Preview</div>
            <div className="prose prose-xs dark:prose-invert max-w-none">
              <ReactMarkdown
                components={{
                  code: CodeBlock,
                  p: ({ children }) => <div className="mb-4 last:mb-0">{children}</div>,
                  strong: ({ children }) => <span className="font-bold">{children}</span>
                }}
              >
                {input}
              </ReactMarkdown>
            </div>
          </div>
        )}

        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask me anything or upload an image..."
          className={cn(
            "w-full p-4 pr-12 bg-transparent border-none focus:ring-0 resize-none min-h-[56px] text-zinc-800 dark:text-zinc-100 placeholder:text-zinc-400 dark:placeholder:text-zinc-500",
            showPreview && "hidden"
          )}
          rows={1}
          disabled={isGenerating}
        />
        
        {showPreview && !input.trim() && (
          <div className="p-12 text-center text-zinc-400 text-sm italic">
            Nothing to preview...
          </div>
        )}

        <div className="flex items-center justify-between px-3 pb-3">
          <div className="flex items-center gap-1">
            <input 
              type="file" 
              ref={fileInputRef} 
              onChange={handleFileChange} 
              accept="image/*" 
              multiple
              className="hidden" 
            />
            <input 
              type="file" 
              ref={docInputRef} 
              onChange={handleFileChange} 
              accept=".pdf,.doc,.docx,.txt,.csv,.json" 
              className="hidden" 
            />
            <button 
              onClick={() => fileInputRef.current?.click()}
              title="Upload image"
              className="p-2 text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-200 hover:bg-zinc-100 dark:hover:bg-zinc-800 rounded-lg transition-colors"
            >
              <ImageIcon className="w-5 h-5" />
            </button>
            <button 
              onClick={() => docInputRef.current?.click()}
              title="Upload document"
              className="p-2 text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-200 hover:bg-zinc-100 dark:hover:bg-zinc-800 rounded-lg transition-colors"
            >
              <Paperclip className="w-5 h-5" />
            </button>
            <button 
              onClick={toggleListening}
              title={isListening ? "Stop listening" : "Voice input"}
              className={cn(
                "p-2 rounded-lg transition-all duration-200",
                isListening 
                  ? "text-red-500 bg-red-50 dark:bg-red-900/20 animate-pulse" 
                  : "text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-200 hover:bg-zinc-100 dark:hover:bg-zinc-800"
              )}
            >
              <Mic className={cn("w-5 h-5", isListening && "fill-current")} />
            </button>
            <button 
              onClick={() => setShowPreview(!showPreview)}
              title={showPreview ? "Edit prompt" : "Preview markdown"}
              className={cn(
                "p-2 rounded-lg transition-all duration-200",
                showPreview 
                  ? "text-black dark:text-white bg-zinc-100 dark:bg-zinc-800" 
                  : "text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-200 hover:bg-zinc-100 dark:hover:bg-zinc-800"
              )}
            >
              {showPreview ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
            </button>
          </div>
          
          <div className="flex items-center gap-3">
            <span className="text-[10px] font-mono text-zinc-400 dark:text-zinc-500">
              {input.length} characters
            </span>
            <button
              onClick={isGenerating ? onStop : handleSubmit}
              title={isGenerating ? "Stop generation" : "Send message"}
              disabled={!isGenerating && (!input.trim() && selectedImages.length === 0 && !selectedImage && !attachedFile)}
              className={cn(
                "p-2 rounded-xl transition-all duration-200",
                isGenerating 
                  ? "bg-red-500 text-white hover:bg-red-600"
                  : (input.trim() || selectedImages.length > 0 || selectedImage || attachedFile)
                    ? "bg-black dark:bg-white text-white dark:text-black hover:scale-105 active:scale-95"
                    : "bg-zinc-100 dark:bg-zinc-800 text-zinc-300 dark:text-zinc-600 cursor-not-allowed"
              )}
            >
              {isGenerating ? <Square className="w-5 h-5 fill-current" /> : <Send className="w-5 h-5" />}
            </button>
          </div>
        </div>
      </div>
      <p className="mt-3 text-center text-[10px] text-zinc-400 dark:text-zinc-600">
        Toxic AI Turbo • Flash Model • Multi-modal
      </p>
    </div>
  );
};

const VoiceCallOverlay: React.FC<{
  isOpen: boolean;
  onClose: () => void;
  settings: AppSettings;
  onSendMessage: (text: string) => void;
  isTyping: boolean;
  streamingContent: string;
  messages: Message[];
}> = ({ isOpen, onClose, settings, onSendMessage, isTyping, streamingContent, messages }) => {
  const [isMuted, setIsMuted] = useState(false);
  const [status, setStatus] = useState<'connecting' | 'connected' | 'listening' | 'speaking'>('connecting');
  const [transcript, setTranscript] = useState('');
  const [error, setError] = useState<string | null>(null);
  const recognitionRef = useRef<any>(null);
  const statusRef = useRef(status);
  const currentAudioRef = useRef<HTMLAudioElement | null>(null);

  useEffect(() => {
    statusRef.current = status;
  }, [status]);

  const companionName = settings.companionName || (settings.companionType === 'girlfriend' ? 'Aria' : settings.companionType === 'boyfriend' ? 'Alex' : 'Sam');

  useEffect(() => {
    if (isOpen) {
      setTimeout(() => setStatus('connected'), 1500);
    }
    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.onend = null;
        recognitionRef.current.stop();
      }
      if (currentAudioRef.current) {
        currentAudioRef.current.pause();
        currentAudioRef.current = null;
      }
    };
  }, [isOpen]);

  useEffect(() => {
    if (status === 'connected' || status === 'listening') {
      startListening();
    }
  }, [status]);

  useEffect(() => {
    if (isTyping || streamingContent) {
      setStatus('speaking');
      if (recognitionRef.current) recognitionRef.current.stop();
    } else if (status === 'speaking') {
      setStatus('listening');
    }
  }, [isTyping, streamingContent]);

  // Auto-play last message if it's from model and we're in a call
  useEffect(() => {
    let isMounted = true;
    if (isOpen && messages.length > 0) {
      const lastMessage = messages[messages.length - 1];
      if (lastMessage.role === 'model' && !isTyping && !streamingContent) {
        const cacheKey = `${settings.voice}_${lastMessage.content}`;
        
        const playAudio = (url: string) => {
          if (!isMounted) return;
          // Stop any currently playing audio
          if (currentAudioRef.current) {
            currentAudioRef.current.pause();
          }

          const audio = new Audio(url);
          currentAudioRef.current = audio;
          
          audio.onplay = () => {
            if (isMounted) setStatus('speaking');
          };
          audio.onended = () => {
            if (isMounted) {
              setStatus('listening');
              currentAudioRef.current = null;
            }
          };
          audio.play().catch(e => {
            console.error('Call audio playback failed:', e);
            if (isMounted) setStatus('listening');
          });
        };

        if (audioCache.has(cacheKey)) {
          playAudio(audioCache.get(cacheKey)!);
        } else {
          generateSpeech(lastMessage.content, settings.voice)
            .then(url => {
              if (isMounted && url) {
                audioCache.set(cacheKey, url);
                playAudio(url);
              }
            })
            .catch(err => console.error('Call speech generation failed:', err));
        }
      }
    }
    return () => {
      isMounted = false;
    };
  }, [messages, isOpen, isTyping, streamingContent]);

  const startListening = () => {
    if (isMuted || status === 'speaking' || !isOpen) return;

    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SpeechRecognition) return;

    if (!recognitionRef.current) {
      const recognition = new SpeechRecognition();
      recognition.continuous = false;
      recognition.interimResults = true;
      recognition.lang = 'en-US';

      recognition.onstart = () => {
        setStatus('listening');
        setError(null);
      };
      recognition.onerror = (event: any) => {
        console.error('Speech recognition error:', event.error);
        if (event.error === 'not-allowed') {
          setError('Microphone access denied. Please enable it in settings.');
        } else {
          setError(`Error: ${event.error}`);
        }
        setStatus('connected');
      };
      recognition.onresult = (event: any) => {
        const current = event.results[event.results.length - 1][0].transcript;
        setTranscript(current);
        if (event.results[event.results.length - 1].isFinal) {
          onSendMessage(current);
          setTranscript('');
        }
      };
      recognition.onend = () => {
        if (statusRef.current !== 'speaking' && isOpen) {
           try { recognition.start(); } catch (e) {}
        }
      };
      recognitionRef.current = recognition;
    }

    try {
      recognitionRef.current.start();
    } catch (e) {}
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-[100] bg-black flex flex-col items-center justify-between p-8 sm:p-12 text-white"
    >
      <div className="flex flex-col items-center gap-4 mt-12">
        <div className="relative">
          <motion.div
            animate={{ scale: [1, 1.1, 1] }}
            transition={{ duration: 2, repeat: Infinity }}
            className={cn(
              "w-32 h-32 sm:w-40 sm:h-40 rounded-full flex items-center justify-center text-4xl sm:text-5xl shadow-2xl",
              settings.companionType === 'girlfriend' ? "bg-red-500/20 text-red-500 border-2 border-red-500/50" :
              settings.companionType === 'boyfriend' ? "bg-blue-500/20 text-blue-500 border-2 border-blue-500/50" :
              "bg-purple-500/20 text-purple-500 border-2 border-purple-500/50"
            )}
          >
            {settings.companionType === 'girlfriend' ? <Heart className="w-16 h-16 fill-current" /> :
             settings.companionType === 'boyfriend' ? <Heart className="w-16 h-16 fill-current" /> :
             <Users className="w-16 h-16" />}
          </motion.div>
          {status === 'speaking' && (
            <div className="absolute -inset-4 border-4 border-white/20 rounded-full animate-ping" />
          )}
        </div>
        <h2 className="text-3xl sm:text-4xl font-black tracking-tighter mt-4">{companionName}</h2>
        <div className="flex items-center gap-2 text-zinc-400 font-bold uppercase tracking-widest text-xs">
          <div className={cn("w-2 h-2 rounded-full", status === 'connecting' ? "bg-zinc-500" : "bg-emerald-500 animate-pulse")} />
          {status === 'connecting' ? 'Connecting...' : status === 'speaking' ? 'Speaking...' : status === 'listening' ? 'Listening...' : 'Connected'}
        </div>
      </div>

      <div className="flex flex-col items-center gap-8 w-full max-w-md">
        {error && (
          <motion.div 
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-red-500/20 border border-red-500/50 text-red-500 px-4 py-2 rounded-xl text-sm font-bold flex items-center gap-2"
          >
            <X className="w-4 h-4" />
            {error}
          </motion.div>
        )}

        {transcript && (
          <motion.p 
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center text-zinc-400 italic text-lg px-4"
          >
            "{transcript}"
          </motion.p>
        )}

        {status === 'speaking' && (
          <div className="flex gap-1 items-end h-12">
            {[...Array(8)].map((_, i) => (
              <motion.div
                key={i}
                animate={{ height: [10, 40, 10] }}
                transition={{ duration: 0.5, repeat: Infinity, delay: i * 0.1 }}
                className="w-1.5 bg-white rounded-full"
              />
            ))}
          </div>
        )}

        <div className="flex items-center gap-8 sm:gap-12 mb-12">
          <button
            onClick={() => setIsMuted(!isMuted)}
            className={cn(
              "w-16 h-16 sm:w-20 sm:h-20 rounded-full flex items-center justify-center transition-all hover:scale-110 active:scale-95",
              isMuted ? "bg-zinc-800 text-zinc-400" : "bg-zinc-800 text-white"
            )}
          >
            {isMuted ? <MicOff className="w-6 h-6 sm:w-8 sm:h-8" /> : <Mic className="w-6 h-6 sm:w-8 sm:h-8" />}
          </button>
          
          <button
            onClick={onClose}
            className="w-20 h-20 sm:w-24 sm:h-24 rounded-full bg-red-600 flex items-center justify-center text-white shadow-2xl shadow-red-600/40 hover:bg-red-700 hover:scale-110 active:scale-95 transition-all"
          >
            <PhoneOff className="w-8 h-8 sm:w-10 sm:h-10" />
          </button>

          <button
            className="w-16 h-16 sm:w-20 sm:h-20 rounded-full bg-zinc-800 flex items-center justify-center text-white hover:scale-110 active:scale-95 transition-all"
          >
            <Volume2 className="w-6 h-6 sm:w-8 sm:h-8" />
          </button>
        </div>
      </div>
    </motion.div>
  );
};

// --- MAIN APP ---

interface AppSettings {
  model: string;
  theme: 'light' | 'dark';
  companionType: 'none' | 'friend' | 'girlfriend' | 'boyfriend';
  companionName: string;
  voice: string;
  autoPlayVoice: boolean;
}

interface ChatSession {
  id: string;
  title: string;
  messages: Message[];
  timestamp: number;
}

export default function App() {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isTyping, setIsTyping] = useState(false);
  const abortControllerRef = useRef<AbortController | null>(null);
  const [streamingContent, setStreamingContent] = useState('');
  const [currentGrounding, setCurrentGrounding] = useState<any>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [showScrollButton, setShowScrollButton] = useState(false);
  const [confirmClear, setConfirmClear] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isCalling, setIsCalling] = useState(false);
  const [settings, setSettings] = useState<AppSettings>({
    model: 'gemini-3-flash-preview',
    theme: 'light',
    companionType: 'none',
    companionName: '',
    voice: 'Puck',
    autoPlayVoice: false
  });
  const scrollRef = useRef<HTMLDivElement>(null);

  // Load initial data
  useEffect(() => {
    try {
      const savedSessions = localStorage.getItem('gemini_copilot_sessions');
      if (savedSessions) {
        const parsed = JSON.parse(savedSessions);
        setSessions(parsed);
      }

      const savedSettings = localStorage.getItem('gemini_copilot_settings');
      if (savedSettings) {
        setSettings(JSON.parse(savedSettings));
      }
    } catch (e) {
      console.error('Failed to load data from localStorage:', e);
    }
  }, []);

  // Save settings
  useEffect(() => {
    try {
      localStorage.setItem('gemini_copilot_settings', JSON.stringify(settings));
    } catch (e) {
      console.error('Failed to save settings to localStorage:', e);
    }
  }, [settings]);

  // Save sessions whenever messages or sessions change
  useEffect(() => {
    if (currentSessionId && messages.length > 0) {
      try {
        setSessions(prevSessions => {
          const sessionExists = prevSessions.find(s => s.id === currentSessionId);
          let updatedSessions: ChatSession[];

          if (sessionExists) {
            updatedSessions = prevSessions.map(s => 
              s.id === currentSessionId ? { ...s, messages, timestamp: Date.now() } : s
            );
          } else {
            const newSession: ChatSession = {
              id: currentSessionId,
              title: messages[0].content.slice(0, 40) + (messages[0].content.length > 40 ? '...' : ''),
              messages,
              timestamp: Date.now()
            };
            updatedSessions = [newSession, ...prevSessions];
          }
          
          localStorage.setItem('gemini_copilot_sessions', JSON.stringify(updatedSessions));
          return updatedSessions;
        });
      } catch (e) {
        console.error('Failed to save sessions to localStorage:', e);
      }
    }
  }, [messages, currentSessionId]);

  const createNewChat = () => {
    setMessages([]);
    setCurrentSessionId(null);
    setShowHistory(false);
  };

  const loadSession = (session: ChatSession) => {
    setMessages(session.messages);
    setCurrentSessionId(session.id);
    setShowHistory(false);
  };

  const deleteSession = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    const updated = sessions.filter(s => s.id !== id);
    setSessions(updated);
    localStorage.setItem('gemini_copilot_sessions', JSON.stringify(updated));
    if (currentSessionId === id) {
      createNewChat();
    }
  };

  const clearAllHistory = () => {
    // Clear current chat state first to prevent the auto-save effect from re-saving it
    setMessages([]);
    setCurrentSessionId(null);
    setSessions([]);
    localStorage.removeItem('gemini_copilot_sessions');
    setShowHistory(false);
    setConfirmClear(false);
  };

  const scrollToBottom = () => {
    if (scrollRef.current) {
      scrollRef.current.scrollTo({
        top: scrollRef.current.scrollHeight,
        behavior: 'smooth'
      });
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, streamingContent]);

  const handleSend = async (content: string, images?: string[]) => {
    // Initialize session ID if it's a new chat
    let sessionId = currentSessionId;
    if (!sessionId) {
      sessionId = Date.now().toString();
      setCurrentSessionId(sessionId);
    }

    const userMessage: Message = { role: 'user', content, images };
    setMessages(prev => [...prev, userMessage]);
    setIsTyping(true);
    setStreamingContent('');
    setCurrentGrounding(null);

    // Set up abort controller
    const controller = new AbortController();
    abortControllerRef.current = controller;

    const contentLower = content.toLowerCase();
    const imageKeywords = ['image', 'picture', 'pic', 'photo', 'illustration', 'sketch', 'drawing', 'portrait'];
    const actionKeywords = ['generate', 'create', 'draw', 'paint', 'make', 'show me', 'produce'];
    
    const hasAction = actionKeywords.some(k => contentLower.includes(k));
    const hasImageNoun = imageKeywords.some(k => contentLower.includes(k));
    
    const isImageRequest = (hasAction && hasImageNoun) || 
                          contentLower.startsWith('draw ') || 
                          contentLower.startsWith('paint ') ||
                          contentLower.includes('generate an image') ||
                          contentLower.includes('create an image');

    try {
      if (isImageRequest && (!images || images.length === 0)) {
        setStreamingContent('Generating your image... 🎨');
        // Use the selected model if it's the image model, otherwise default to image model
        const imageModel = settings.model.includes('image') ? settings.model : 'gemini-2.5-flash-image';
        const generatedImageUrl = await generateImage(content, imageModel);
        if (generatedImageUrl) {
          setMessages(prev => [...prev, { 
            role: 'model', 
            content: `I've generated this image based on your request: "${content}"`,
            image: generatedImageUrl,
            isImage: true
          }]);
        } else {
          setMessages(prev => [...prev, { 
            role: 'model', 
            content: 'I failed to generate the image. Please try a different prompt.' 
          }]);
        }
        setStreamingContent('');
      } else {
        let fullResponse = '';
        let lastGrounding = null;
        
        const stream = sendMessageStream(
          content, 
          messages, 
          settings.model, 
          images, 
          controller.signal,
          { type: settings.companionType, name: settings.companionName }
        );
        
        for await (const chunk of stream) {
          fullResponse += chunk.text;
          setStreamingContent(fullResponse);
          if (chunk.groundingMetadata) {
            lastGrounding = chunk.groundingMetadata;
            setCurrentGrounding(lastGrounding);
          }
        }

        if (!controller.signal.aborted) {
          setMessages(prev => [...prev, { 
            role: 'model', 
            content: fullResponse,
            groundingMetadata: lastGrounding
          }]);

          // Aggressive Pre-fetch for Voice
          const cacheKey = `${settings.voice}_${fullResponse}`;
          if (!audioCache.has(cacheKey)) {
            generateSpeech(fullResponse, settings.voice)
              .then(url => {
                if (url) audioCache.set(cacheKey, url);
              })
              .catch(err => console.error('Background pre-fetch failed:', err));
          }
        }
        setStreamingContent('');
        setCurrentGrounding(null);
      }
    } catch (error: any) {
      if (error instanceof Error && error.name === 'AbortError') {
        console.log('Generation stopped by user');
      } else {
        console.error('Error sending message:', error);
        
        const errorMessage = getDetailedErrorMessage(error, settings.model);

        setMessages(prev => [...prev, { 
          role: 'model', 
          content: errorMessage
        }]);
      }
    } finally {
      setIsTyping(false);
      abortControllerRef.current = null;
    }
  };

  const handleStop = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
      setIsTyping(false);
      setStreamingContent('');
    }
  };

  return (
    <div className={cn(
      "flex h-screen font-sans overflow-hidden w-full max-w-full transition-colors duration-300",
      settings.theme === 'dark' ? "dark bg-zinc-950 text-zinc-100" : "bg-[#F5F5F5] text-zinc-900"
    )}>
      {/* Sidebar */}
      <AnimatePresence mode="wait">
        {isSidebarOpen && (
          <motion.aside 
            initial={{ width: 0, opacity: 0 }}
            animate={{ width: 280, opacity: 1 }}
            exit={{ width: 0, opacity: 0 }}
            className="hidden lg:flex flex-col bg-white dark:bg-zinc-900 border-r border-zinc-200 dark:border-zinc-800 overflow-hidden shrink-0"
          >
            <div className="p-4 flex flex-col h-full gap-4">
              <button 
                onClick={createNewChat}
                title="Start a new conversation"
                className="flex items-center gap-3 w-full p-3 rounded-xl border border-zinc-200 dark:border-zinc-800 hover:bg-zinc-50 dark:hover:bg-zinc-800 transition-all font-bold text-sm"
              >
                <div className="p-1.5 bg-black dark:bg-white dark:text-black text-white rounded-lg">
                  <Zap className="w-4 h-4 fill-current" />
                </div>
                New Chat
              </button>

              <div className="flex-1 overflow-y-auto custom-scrollbar flex flex-col gap-1">
                <div className="px-2 py-2 text-[10px] font-bold uppercase tracking-widest text-zinc-400">Recent Chats</div>
                {sessions.length === 0 ? (
                  <div className="px-4 py-8 text-center text-zinc-400 text-xs italic">No history yet</div>
                ) : (
                  [...sessions].sort((a, b) => b.timestamp - a.timestamp).map((session) => (
                    <div
                      key={session.id}
                      onClick={() => loadSession(session)}
                      title={`Load session: ${session.title}`}
                      className={cn(
                        "w-full flex items-center justify-between p-3 rounded-xl transition-all text-left group cursor-pointer",
                        currentSessionId === session.id
                          ? "bg-zinc-100 dark:bg-zinc-800 text-black dark:text-white"
                          : "hover:bg-zinc-50 dark:hover:bg-zinc-800/50 text-zinc-600 dark:text-zinc-400"
                      )}
                    >
                      <div className="flex items-center gap-3 overflow-hidden">
                        <History className="w-4 h-4 shrink-0 opacity-50" />
                        <span className="font-medium truncate text-xs">{session.title}</span>
                      </div>
                      <button 
                        onClick={(e) => deleteSession(session.id, e)}
                        title="Delete this session"
                        className="p-1.5 opacity-0 group-hover:opacity-100 hover:bg-red-50 dark:hover:bg-red-900/20 text-zinc-300 hover:text-red-500 rounded-md transition-all"
                      >
                        <X className="w-3 h-3" />
                      </button>
                    </div>
                  ))
                )}
              </div>

              <div className="pt-4 border-t border-zinc-100 dark:border-zinc-800 flex flex-col gap-1">
                <button 
                  onClick={() => setShowSettings(true)}
                  title="Settings"
                  className="flex items-center gap-3 w-full p-3 rounded-xl hover:bg-zinc-50 dark:hover:bg-zinc-800 transition-all text-sm font-medium text-zinc-600 dark:text-zinc-400"
                >
                  <Settings className="w-5 h-5" />
                  Settings
                </button>
                <button 
                  title="Profile"
                  className="flex items-center gap-3 w-full p-3 rounded-xl hover:bg-zinc-50 dark:hover:bg-zinc-800 transition-all text-sm font-medium text-zinc-600 dark:text-zinc-400"
                >
                  <User className="w-5 h-5" />
                  Profile
                </button>
              </div>
            </div>
          </motion.aside>
        )}
      </AnimatePresence>

      {/* Main Content */}
      <main className="flex-1 flex flex-col relative min-w-0 overflow-hidden">
        {/* Header */}
        <header className="h-16 flex items-center justify-between px-4 sm:px-6 bg-white/80 dark:bg-zinc-900/80 backdrop-blur-md border-b border-zinc-200 dark:border-zinc-800 z-10 w-full shrink-0">
          <div className="flex items-center gap-3 overflow-hidden">
            <button 
              onClick={() => setIsSidebarOpen(!isSidebarOpen)}
              title={isSidebarOpen ? "Close sidebar" : "Open sidebar"}
              className="hidden lg:flex p-2 hover:bg-zinc-100 dark:hover:bg-zinc-800 rounded-xl transition-colors text-zinc-500"
            >
              <LayoutGrid className="w-5 h-5" />
            </button>
            <div className="flex items-center gap-1.5 sm:gap-2 overflow-hidden">
              <span className="font-bold text-base sm:text-lg tracking-tight truncate bg-clip-text text-transparent bg-gradient-to-r from-red-600 to-orange-500 dark:from-red-400 dark:to-orange-300">Toxic AI</span>
              <div className="flex items-center gap-1 px-1.5 py-0.5 bg-emerald-50 dark:bg-emerald-900/20 text-[9px] sm:text-[10px] font-bold uppercase rounded border border-emerald-100 dark:border-emerald-800 text-emerald-600 dark:text-emerald-400 shrink-0">
                <Zap className="w-2 h-2 sm:w-2.5 sm:h-2.5 fill-current" />
                <span>{settings.model.includes('pro') ? 'Pro 3.1' : 'Flash 1.5'}</span>
              </div>
              {settings.companionType !== 'none' && (
                <div className="flex items-center gap-1 sm:gap-2">
                  <div className={cn(
                    "flex items-center gap-1 px-1.5 py-0.5 text-[9px] sm:text-[10px] font-bold uppercase rounded border shrink-0 animate-in zoom-in duration-300",
                    settings.companionType === 'girlfriend' ? "bg-red-50 dark:bg-red-900/20 border-red-100 dark:border-red-800 text-red-600 dark:text-red-400" :
                    settings.companionType === 'boyfriend' ? "bg-blue-50 dark:bg-blue-900/20 border-blue-100 dark:border-blue-800 text-blue-600 dark:text-blue-400" :
                    "bg-purple-50 dark:bg-purple-900/20 border-purple-100 dark:border-purple-800 text-purple-600 dark:text-purple-400"
                  )}>
                    <Heart className={cn("w-2 h-2 sm:w-2.5 sm:h-2.5 fill-current", settings.companionType === 'friend' && "hidden")} />
                    <Users className={cn("w-2 h-2 sm:w-2.5 sm:h-2.5", settings.companionType !== 'friend' && "hidden")} />
                    <span>{settings.companionName || settings.companionType}</span>
                    <div className="flex items-center gap-0.5 ml-1">
                      <div className="w-1 h-1 rounded-full bg-current animate-pulse" />
                      <span className="text-[7px] opacity-70">Online</span>
                    </div>
                  </div>
                  <button
                    onClick={() => {
                      // Mark as upcoming
                      alert('Calling feature is coming soon! Stay tuned for updates.');
                    }}
                    title={`Call ${settings.companionName || settings.companionType} (Upcoming)`}
                    className="p-1.5 sm:p-2 bg-zinc-200 dark:bg-zinc-800 text-zinc-400 rounded-full hover:scale-110 active:scale-95 transition-all shadow-lg cursor-not-allowed"
                  >
                    <div className="relative">
                      <Phone className="w-3 h-3 sm:w-4 sm:h-4 fill-current" />
                      <div className="absolute -top-1 -right-1 w-1.5 h-1.5 bg-orange-500 rounded-full border border-white dark:border-zinc-900" />
                    </div>
                  </button>
                </div>
              )}
            </div>
          </div>
          <div className="flex items-center gap-2 sm:gap-4">
            <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 bg-zinc-100 dark:bg-zinc-800 rounded-full text-[10px] sm:text-xs font-medium text-zinc-500 dark:text-zinc-400">
              <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse" />
              Online
            </div>
            <button 
              onClick={() => setShowHistory(true)}
              title="Chat History"
              className="lg:hidden p-2 text-zinc-400 hover:text-zinc-900 dark:hover:text-white"
            >
              <History className="w-5 h-5" />
            </button>
            <button 
              onClick={() => setShowSettings(true)}
              title="Settings"
              className="lg:hidden p-2 text-zinc-400 hover:text-zinc-900 dark:hover:text-white"
            >
              <Settings className="w-5 h-5" />
            </button>
            <button 
              onClick={createNewChat}
              title="Start a new conversation"
              className="px-3 py-1.5 sm:px-4 sm:py-1.5 bg-black dark:bg-white text-white dark:text-black text-xs sm:text-sm font-medium rounded-full hover:bg-zinc-800 dark:hover:bg-zinc-200 transition-all hover:scale-105 active:scale-95 shadow-md shadow-black/5 whitespace-nowrap"
            >
              New Chat
            </button>
          </div>
        </header>

        {/* Chat Area */}
        <div 
          ref={scrollRef}
          onScroll={(e) => {
            const target = e.currentTarget;
            const isAtBottom = target.scrollHeight - target.scrollTop <= target.clientHeight + 100;
            setShowScrollButton(!isAtBottom);
          }}
          className="flex-1 overflow-y-auto px-2 sm:px-4 py-6 md:px-8 scroll-smooth min-w-0 relative"
        >
          <AnimatePresence>
            {showScrollButton && (
              <motion.button
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 10 }}
                onClick={scrollToBottom}
                className="fixed bottom-32 right-8 z-20 p-3 bg-black dark:bg-white text-white dark:text-black rounded-full shadow-2xl hover:scale-110 active:scale-90 transition-transform"
              >
                <ChevronDown className="w-5 h-5" />
              </motion.button>
            )}
          </AnimatePresence>
          <div className="max-w-3xl mx-auto w-full">
            {messages.length === 0 && (
              <motion.div 
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                className="flex flex-col items-center justify-center min-h-[60vh] text-center px-4"
              >
                <div className="w-16 h-16 sm:w-20 sm:h-20 bg-black dark:bg-white dark:text-black rounded-[1.5rem] sm:rounded-[2rem] flex items-center justify-center text-white mb-2 shadow-2xl rotate-3 hover:rotate-0 transition-transform duration-500">
                  <Zap className="w-8 h-8 sm:w-10 sm:h-10 fill-current" />
                </div>
                <div className="mb-6 sm:mb-8">
                  <span className="text-xl sm:text-2xl font-black tracking-tighter bg-clip-text text-transparent bg-gradient-to-r from-red-600 to-orange-500 dark:from-red-400 dark:to-orange-300">
                    Toxic AI
                  </span>
                </div>
                <h1 className="text-3xl sm:text-5xl font-black tracking-tighter mb-4 bg-clip-text text-transparent bg-gradient-to-b from-black to-zinc-600 dark:from-white dark:to-zinc-500">
                  Turbocharged AI.
                </h1>
                <p className="text-zinc-500 dark:text-zinc-400 max-w-md mx-auto mb-8 sm:mb-12 text-base sm:text-lg font-medium">
                  Experience lightning-fast responses, image generation, and visual analysis.
                </p>
                
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 sm:gap-4 w-full max-w-2xl">
                  {[
                    { text: "Generate an image of a futuristic city", icon: <ImageIcon className="w-4 h-4" /> },
                    { text: "Analyze this image for me", icon: <LayoutGrid className="w-4 h-4" /> },
                    { 
                      text: "Voice Call with my AI", 
                      icon: <Phone className="w-4 h-4" />, 
                      action: () => alert('Calling feature is coming soon! Stay tuned for updates.'),
                      isUpcoming: true
                    },
                    { text: "Summarize the latest tech news", icon: <Sparkles className="w-4 h-4" /> },
                    { text: "Write a high-performance API", icon: <Zap className="w-4 h-4" /> }
                  ].map((suggestion, i) => (
                    <button
                      key={i}
                      onClick={() => suggestion.action ? suggestion.action() : handleSend(suggestion.text)}
                      title={suggestion.isUpcoming ? "Coming Soon" : `Try: "${suggestion.text}"`}
                      className={cn(
                        "p-4 sm:p-5 bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-2xl sm:rounded-3xl text-left hover:border-black dark:hover:border-white hover:shadow-xl transition-all group relative overflow-hidden",
                        suggestion.isUpcoming && "opacity-60 grayscale-[0.5]"
                      )}
                    >
                      <div className="absolute top-0 right-0 p-3 text-zinc-100 dark:text-zinc-800 group-hover:text-zinc-200 dark:group-hover:text-zinc-700 transition-colors">
                        {suggestion.icon}
                      </div>
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-zinc-900 dark:text-white font-bold block text-sm sm:text-base">{suggestion.text}</span>
                        {suggestion.isUpcoming && (
                          <span className="px-1.5 py-0.5 bg-orange-100 dark:bg-orange-900/30 text-orange-600 dark:text-orange-400 text-[8px] font-black uppercase rounded">Soon</span>
                        )}
                      </div>
                      <div className="text-[10px] sm:text-xs text-zinc-400 dark:text-zinc-500 font-medium">
                        {suggestion.isUpcoming ? "Upcoming Feature" : "Try Turbo speed →"}
                      </div>
                    </button>
                  ))}
                </div>
              </motion.div>
            )}

            <div className="space-y-2">
              {messages.map((msg, i) => (
                <MessageBubble key={i} message={msg} voice={settings.voice} autoPlay={settings.autoPlayVoice && i === messages.length - 1} />
              ))}
              {isTyping && streamingContent && (
                <MessageBubble 
                  message={{ role: 'model', content: streamingContent, groundingMetadata: currentGrounding }} 
                  isStreaming={true} 
                  voice={settings.voice}
                />
              )}
              {isTyping && !streamingContent && (
                <div className="flex justify-start mb-6 animate-in fade-in slide-in-from-left-2 duration-300">
                  <div className="bg-white dark:bg-zinc-900 border border-black/5 dark:border-white/5 px-4 py-3 rounded-2xl rounded-tl-none shadow-sm flex items-center gap-3">
                    <div className="flex gap-1">
                      <div className="w-1.5 h-1.5 bg-zinc-300 dark:bg-zinc-600 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                      <div className="w-1.5 h-1.5 bg-zinc-400 dark:bg-zinc-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                      <div className="w-1.5 h-1.5 bg-zinc-500 dark:bg-zinc-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                    </div>
                    <span className="text-[10px] font-bold uppercase tracking-widest text-zinc-400">
                      {settings.companionType !== 'none' ? `${settings.companionName || settings.companionType} is thinking...` : 'Toxic AI is thinking...'}
                    </span>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Input Area */}
        <div className="bg-gradient-to-t from-[#F5F5F5] dark:from-zinc-950 via-[#F5F5F5] dark:via-zinc-950 to-transparent pt-8">
          <InputArea onSend={handleSend} onStop={handleStop} isGenerating={isTyping} />
        </div>
      </main>

      {/* History Modal */}
      <AnimatePresence>
        {showHistory && (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setShowHistory(false)}
              className="absolute inset-0 bg-black/40 backdrop-blur-sm"
            />
            <motion.div
              initial={{ opacity: 0, scale: 0.95, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 20 }}
              className="relative w-full max-w-md bg-white dark:bg-zinc-900 rounded-3xl shadow-2xl overflow-hidden border border-zinc-200 dark:border-zinc-800 flex flex-col max-h-[80vh]"
            >
              <div className="p-6 border-b border-zinc-100 dark:border-zinc-800 flex items-center justify-between shrink-0">
                <h2 className="text-xl font-bold flex items-center gap-2">
                  <History className="w-5 h-5" />
                  Chat History
                </h2>
                <button 
                  onClick={() => setShowHistory(false)}
                  title="Close history"
                  className="p-2 hover:bg-zinc-100 dark:hover:bg-zinc-800 rounded-full transition-colors"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
              
              <div className="flex-1 overflow-y-auto p-4 space-y-2 custom-scrollbar">
                {sessions.length === 0 ? (
                  <div className="flex flex-col items-center justify-center py-12 text-zinc-400">
                    <History className="w-12 h-12 mb-4 opacity-20" />
                    <p className="font-medium">No history yet</p>
                  </div>
                ) : (
                  [...sessions].sort((a, b) => b.timestamp - a.timestamp).map((session) => (
                    <div
                      key={session.id}
                      onClick={() => loadSession(session)}
                      title={`Load session: ${session.title}`}
                      className={cn(
                        "w-full flex items-center justify-between p-4 rounded-2xl border transition-all text-left group cursor-pointer",
                        currentSessionId === session.id
                          ? "bg-black text-white border-black dark:bg-white dark:text-black dark:border-white"
                          : "bg-white dark:bg-zinc-800 border-zinc-100 dark:border-zinc-700 hover:border-zinc-300 dark:hover:border-zinc-500"
                      )}
                    >
                      <div className="flex flex-col overflow-hidden pr-4">
                        <span className="font-bold truncate text-sm">{session.title}</span>
                        <span className={cn("text-[10px]", currentSessionId === session.id ? "opacity-60" : "text-zinc-400")}>
                          {new Date(session.timestamp).toLocaleString()}
                        </span>
                      </div>
                      <button 
                        onClick={(e) => deleteSession(session.id, e)}
                        title="Delete this session"
                        className={cn(
                          "p-2 rounded-lg transition-colors shrink-0",
                          currentSessionId === session.id 
                            ? "hover:bg-white/20 text-white/60 hover:text-white" 
                            : "hover:bg-zinc-100 dark:hover:bg-zinc-700 text-zinc-400 hover:text-red-500"
                        )}
                      >
                        <X className="w-4 h-4" />
                      </button>
                    </div>
                  ))
                )}
              </div>

              <div className="p-6 bg-zinc-50 dark:bg-zinc-900/50 border-t border-zinc-100 dark:border-zinc-800 flex gap-3 shrink-0">
                {confirmClear ? (
                  <button 
                    onClick={clearAllHistory}
                    className="flex-1 py-3 bg-red-500 text-white font-bold rounded-2xl hover:bg-red-600 transition-all animate-pulse"
                  >
                    Confirm Clear?
                  </button>
                ) : (
                  <button 
                    onClick={() => setConfirmClear(true)}
                    title="Delete all saved conversations"
                    disabled={sessions.length === 0}
                    className="flex-1 py-3 border border-zinc-200 dark:border-zinc-700 text-zinc-600 dark:text-zinc-400 font-bold rounded-2xl hover:bg-zinc-100 dark:hover:bg-zinc-800 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                  >
                    Clear All
                  </button>
                )}
                <button 
                  onClick={() => {
                    setShowHistory(false);
                    setConfirmClear(false);
                  }}
                  title="Close history"
                  className="flex-1 py-3 bg-black dark:bg-white text-white dark:text-black font-bold rounded-2xl hover:scale-[1.02] active:scale-[0.98] transition-all"
                >
                  {confirmClear ? "Cancel" : "Done"}
                </button>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>

      {/* Voice Call Overlay */}
      <AnimatePresence>
        {isCalling && (
          <VoiceCallOverlay 
            isOpen={isCalling} 
            onClose={() => setIsCalling(false)} 
            settings={settings}
            onSendMessage={handleSend}
            isTyping={isTyping}
            streamingContent={streamingContent}
            messages={messages}
          />
        )}
      </AnimatePresence>

      {/* Settings Modal */}
      <AnimatePresence>
        {showSettings && (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setShowSettings(false)}
              className="absolute inset-0 bg-black/40 backdrop-blur-sm"
            />
            <motion.div
              initial={{ opacity: 0, scale: 0.95, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 20 }}
              className="relative w-full max-w-md max-h-[90vh] flex flex-col bg-white dark:bg-zinc-900 rounded-3xl shadow-2xl overflow-hidden border border-zinc-200 dark:border-zinc-800"
            >
              <div className="p-6 border-b border-zinc-100 dark:border-zinc-800 flex items-center justify-between">
                <h2 className="text-xl font-bold flex items-center gap-2">
                  <Settings className="w-5 h-5" />
                  Settings
                </h2>
                <button 
                  onClick={() => setShowSettings(false)}
                  title="Close settings"
                  className="p-2 hover:bg-zinc-100 dark:hover:bg-zinc-800 rounded-full transition-colors"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
              
              <div className="p-6 space-y-8 overflow-y-auto flex-1 custom-scrollbar">
                {/* Model Selection */}
                <div className="space-y-3">
                  <label className="text-xs font-bold uppercase tracking-wider text-zinc-400">AI Model</label>
                  <div className="grid grid-cols-1 gap-2">
                    {[
                      { id: 'gemini-3-flash-preview', name: 'Gemini 3 Flash', desc: 'Fastest, great for most tasks' },
                      { id: 'gemini-3.1-pro-preview', name: 'Gemini 3.1 Pro', desc: 'Advanced reasoning & complex tasks' },
                      { id: 'gemini-2.5-flash-image', name: 'Nano Banana (Image)', desc: 'Specialized for high-quality image generation' }
                    ].map((m) => (
                      <button
                        key={m.id}
                        onClick={() => setSettings(s => ({ ...s, model: m.id }))}
                        title={`Switch to ${m.name}`}
                        className={cn(
                          "flex flex-col items-start p-4 rounded-2xl border transition-all text-left",
                          settings.model === m.id 
                            ? "bg-black text-white border-black dark:bg-white dark:text-black dark:border-white" 
                            : "bg-white dark:bg-zinc-800 border-zinc-200 dark:border-zinc-700 hover:border-zinc-400"
                        )}
                      >
                        <span className="font-bold">{m.name}</span>
                        <span className={cn("text-xs", settings.model === m.id ? "opacity-70" : "text-zinc-500")}>{m.desc}</span>
                      </button>
                    ))}
                  </div>
                </div>

                {/* AI Companion Section */}
                <div className="space-y-4">
                  <label className="text-xs font-bold uppercase tracking-wider text-zinc-400 flex items-center gap-2">
                    <Heart className="w-3 h-3 text-red-500" />
                    AI Companion
                  </label>
                  
                  <div className="grid grid-cols-2 gap-2">
                    {[
                      { id: 'none', name: 'None', icon: <X className="w-4 h-4" /> },
                      { id: 'friend', name: 'Friend', icon: <Users className="w-4 h-4" /> },
                      { id: 'girlfriend', name: 'Girlfriend', icon: <Heart className="w-4 h-4 fill-current text-red-500" /> },
                      { id: 'boyfriend', name: 'Boyfriend', icon: <Heart className="w-4 h-4 fill-current text-blue-500" /> }
                    ].map((type) => (
                      <button
                        key={type.id}
                        onClick={() => {
                          const newType = type.id as any;
                          setSettings(s => ({ 
                            ...s, 
                            companionType: newType,
                            voice: newType === 'girlfriend' ? 'Zephyr' : newType === 'boyfriend' ? 'Charon' : s.voice
                          }));
                          if (type.id === 'none') {
                            setMessages([]);
                            setCurrentSessionId(null);
                          }
                        }}
                        className={cn(
                          "flex items-center gap-2 p-3 rounded-xl border transition-all text-sm font-medium",
                          settings.companionType === type.id 
                            ? "bg-black text-white border-black dark:bg-white dark:text-black dark:border-white" 
                            : "bg-white dark:bg-zinc-800 border-zinc-200 dark:border-zinc-700 hover:border-zinc-400"
                        )}
                      >
                        {type.icon}
                        {type.name}
                      </button>
                    ))}
                  </div>

                  {settings.companionType !== 'none' && (
                    <div className="space-y-2 animate-in fade-in slide-in-from-top-2 duration-300">
                      <label className="text-[10px] font-bold uppercase tracking-wider text-zinc-500">Companion Name</label>
                      <div className="flex gap-2">
                        <input 
                          type="text"
                          value={settings.companionName}
                          onChange={(e) => setSettings(s => ({ ...s, companionName: e.target.value }))}
                          placeholder={`Enter ${settings.companionType}'s name...`}
                          className="flex-1 p-3 bg-zinc-100 dark:bg-zinc-800 border-none rounded-xl text-sm focus:ring-2 focus:ring-black dark:focus:ring-white transition-all"
                        />
                        <button
                          onClick={() => {
                            alert('Calling feature is coming soon! Stay tuned for updates.');
                          }}
                          className="p-3 bg-zinc-200 dark:bg-zinc-800 text-zinc-400 rounded-xl hover:scale-105 active:scale-95 transition-all cursor-not-allowed"
                          title="Voice Call (Upcoming)"
                        >
                          <div className="relative">
                            <Phone className="w-5 h-5 fill-current" />
                            <div className="absolute -top-1 -right-1 w-2 h-2 bg-orange-500 rounded-full border-2 border-white dark:border-zinc-900" />
                          </div>
                        </button>
                      </div>
                    </div>
                  )}
                </div>

                {/* Voice Selection */}
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <label className="text-xs font-bold uppercase tracking-wider text-zinc-400">AI Voice</label>
                    <button 
                      onClick={() => setSettings(s => ({ ...s, autoPlayVoice: !s.autoPlayVoice }))}
                      className={cn(
                        "flex items-center gap-2 px-2 py-1 rounded-lg text-[10px] font-bold uppercase transition-all",
                        settings.autoPlayVoice 
                          ? "bg-emerald-500 text-white" 
                          : "bg-zinc-100 dark:bg-zinc-800 text-zinc-400"
                      )}
                    >
                      {settings.autoPlayVoice ? 'Auto-play ON' : 'Auto-play OFF'}
                    </button>
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    {[
                      { id: 'Zephyr', name: 'Sweet Female', icon: <Heart className="w-3 h-3 fill-current text-pink-500" /> },
                      { id: 'Puck', name: 'Cute Female', icon: <Sparkles className="w-3 h-3" /> },
                      { id: 'Kore', name: 'Soft Female', icon: <Users className="w-3 h-3" /> },
                      { id: 'Charon', name: 'Deep Male', icon: <User className="w-3 h-3" /> },
                      { id: 'Fenrir', name: 'Bold Male', icon: <Zap className="w-3 h-3" /> }
                    ].map((v) => (
                      <button
                        key={v.id}
                        onClick={() => setSettings(s => ({ ...s, voice: v.id }))}
                        className={cn(
                          "flex items-center gap-2 p-3 rounded-xl border transition-all text-sm font-medium",
                          settings.voice === v.id 
                            ? "bg-black text-white border-black dark:bg-white dark:text-black dark:border-white" 
                            : "bg-white dark:bg-zinc-800 border-zinc-200 dark:border-zinc-700 hover:border-zinc-400"
                        )}
                      >
                        {v.icon}
                        {v.name}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Theme Selection */}
                <div className="space-y-3">
                  <label className="text-xs font-bold uppercase tracking-wider text-zinc-400">Appearance</label>
                  <div className="flex p-1 bg-zinc-100 dark:bg-zinc-800 rounded-xl">
                    {['light', 'dark'].map((t) => (
                      <button
                        key={t}
                        onClick={() => setSettings(s => ({ ...s, theme: t as any }))}
                        title={`Switch to ${t} mode`}
                        className={cn(
                          "flex-1 py-2 text-sm font-bold rounded-lg capitalize transition-all",
                          settings.theme === t 
                            ? "bg-white dark:bg-zinc-700 shadow-sm text-black dark:text-white" 
                            : "text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300"
                        )}
                      >
                        {t}
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              <div className="p-6 bg-zinc-50 dark:bg-zinc-900/50 border-t border-zinc-100 dark:border-zinc-800">
                <button 
                  onClick={() => setShowSettings(false)}
                  title="Save and close settings"
                  className="w-full py-3 bg-black dark:bg-white text-white dark:text-black font-bold rounded-2xl hover:scale-[1.02] active:scale-[0.98] transition-all"
                >
                  Done
                </button>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </div>
  );
}
