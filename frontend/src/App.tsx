import { useState, useRef, useEffect } from "react";
import axios from "axios";
import ReactMarkdown from "react-markdown";


const API = "https://flynn-rag.onrender.com";

interface Source {
  filename: string;
  page: number;
  relevance: number;
  content: string;
  uploaded_at?: string;
  version?: number;
  is_latest?: boolean;
}

interface DocRecord {
  filename: string;
  uploaded_at: string;
  chunk_count: number;
  version?: number;
  is_latest?: boolean;
}

interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  query_type?: string;
}

const EXAMPLE_QUERIES = [
  "What is the minimum TPO membrane thickness?",
  "Compare TPO and EPDM membranes",
  "What are the fall protection requirements?",
  "What roofing system for a building in Winnipeg?",
  "What is the minimum R-value for Climate Zone 7?",
  "What are the seam welding requirements for TPO?",
];

// ── Format ISO date string ─────────────────────────────────────────────────────
function formatDate(iso: string): string {
  if (!iso) return "";
  try {
    return new Date(iso).toLocaleString("en-CA", {
      month: "short",
      day: "numeric",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return iso;
  }
}

// ── One collapsible card per document; excerpts shown inline ──────────────────
function DocumentSourceGroup({ filename, passages }: { filename: string; passages: Source[] }) {
  const [open, setOpen] = useState(false);
  const [showAll, setShowAll] = useState(false);
  const label = filename.replace(/\.[^.]+$/, "").replace(/_/g, " ");
  const count = passages.length;
  const EXCERPT_LIMIT = 3;
  const visiblePassages = showAll ? passages : passages.slice(0, EXCERPT_LIMIT);
  const hiddenCount = count - EXCERPT_LIMIT;

  return (
    <div className="border border-green-200 rounded-lg overflow-hidden mb-2">
      {/* Header — click to expand/collapse */}
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-4 py-2 bg-green-50 hover:bg-green-100 transition-colors text-left cursor-pointer"
      >
        <span className="text-green-800 text-sm font-medium truncate">
          📋 {label}
          {passages[0]?.version && passages[0].version > 1 && (
            <span className="ml-1.5 bg-blue-100 text-blue-700 text-xs px-1.5 py-0.5 rounded-full font-normal">v{passages[0].version}</span>
          )}
          {passages[0]?.is_latest === false && (
            <span className="ml-1.5 bg-amber-100 text-amber-700 text-xs px-1.5 py-0.5 rounded-full font-normal">old</span>
          )}
        </span>
        <div className="flex items-center gap-3 ml-2 shrink-0">
          <span className="bg-green-700 text-white text-xs px-2 py-0.5 rounded-full">
            {count} {count === 1 ? "excerpt" : "excerpts"}
          </span>
          <span className="text-green-600 text-xs">{open ? "▲" : "▼"}</span>
        </div>
      </button>

      {/* Excerpts — visible when card is open */}
      {open && (
        <div className="bg-white border-t border-green-100 px-4 py-3 space-y-4">
          {visiblePassages.map((source, i) => (
            <div key={i}>
              <p className="text-green-700 text-xs font-semibold mb-1">
                Excerpt {i + 1} — <span className="text-gray-500 font-normal">Page {source.page}</span>
              </p>
              <div className="border-l-4 border-amber-500 pl-3">
                <p className="text-gray-700 text-sm leading-relaxed">{source.content}</p>
              </div>
              {source.uploaded_at && i === visiblePassages.length - 1 && (
                <p className="text-gray-400 text-xs mt-2">
                  Document uploaded: {formatDate(source.uploaded_at)}
                </p>
              )}
            </div>
          ))}
          {hiddenCount > 0 && (
            <button
              onClick={() => setShowAll(!showAll)}
              className="text-green-700 hover:text-green-900 text-xs font-medium cursor-pointer"
            >
              {showAll ? "Show less" : `Show ${hiddenCount} more`}
            </button>
          )}
        </div>
      )}
    </div>
  );
}

// ── Share bar ──────────────────────────────────────────────────────────────────
function ShareBar({ question, answer }: { question: string; answer: string }) {
  const [copied, setCopied] = useState(false);
  const shareText = `Q: ${question}\n\nA: ${answer}\n\nPowered by DocuMind`;
  const encoded = encodeURIComponent(shareText);

  const handleCopy = () => {
    navigator.clipboard.writeText(shareText).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  return (
    <div className="flex items-center gap-4 mt-3 pt-3 border-t border-gray-800">
      <span className="text-gray-500 text-xs">Share on:</span>
      <a
        href={`https://wa.me/?text=${encoded}`}
        target="_blank"
        rel="noreferrer"
        className="text-lg hover:scale-110 transition-transform cursor-pointer"
        title="Share on WhatsApp"
      >
        {"📱"}
      </a>
      <a
        href={`mailto:?subject=DocuMind Answer&body=${encoded}`}
        target="_blank"
        rel="noreferrer"
        className="text-lg hover:scale-110 transition-transform cursor-pointer"
        title="Share via Email"
      >
        {"✉️"}
      </a>
      <button
        onClick={handleCopy}
        className="ml-auto flex items-center gap-1.5 text-xs text-gray-400 border border-gray-700 hover:border-green-600 hover:text-green-400 px-3 py-1 rounded-full transition-colors cursor-pointer"
      >
        {copied ? "✅ Copied" : "📋 Copy"}
      </button>
    </div>
  );
}

// ── Message bubble ─────────────────────────────────────────────────────────────
function MessageBubble({
  message,
  prevQuestion,
}: {
  message: Message;
  prevQuestion?: string;
}) {
  const isUser = message.role === "user";

  if (isUser) {
    return (
      <div className="flex justify-end mb-4">
        <div className="bg-green-700 text-white px-4 py-3 rounded-2xl rounded-tr-sm max-w-xl text-sm leading-relaxed" style={{ whiteSpace: "pre-wrap" }}>
          {message.content}
        </div>
      </div>
    );
  }

  // Group sources by filename for deduplication (#9)
  const groupedSources: Record<string, Source[]> = {};
  for (const source of message.sources ?? []) {
    if (!groupedSources[source.filename]) groupedSources[source.filename] = [];
    groupedSources[source.filename].push(source);
  }

  return (
    <div className="flex justify-start mb-6">
      <div className="max-w-3xl w-full">
        <div className="flex items-center gap-2 mb-2">
          <svg viewBox="0 0 56 70" width="22" height="22" xmlns="http://www.w3.org/2000/svg">
            <rect x="0" y="0" width="56" height="70" rx="5" fill="#1B4332"/>
            <polygon points="38,0 56,18 38,18" fill="#2D6A4F"/>
            <polyline points="38,0 56,18 38,18 38,0" fill="none" stroke="#52B788" strokeWidth="0.8"/>
            <line x1="8" y1="28" x2="30" y2="28" stroke="#52B788" strokeWidth="1.3" strokeLinecap="round"/>
            <line x1="8" y1="37" x2="26" y2="37" stroke="#52B788" strokeWidth="1.3" strokeLinecap="round"/>
            <line x1="8" y1="46" x2="32" y2="46" stroke="#52B788" strokeWidth="1.3" strokeLinecap="round"/>
            <line x1="8" y1="55" x2="22" y2="55" stroke="#52B788" strokeWidth="1.3" strokeLinecap="round"/>
            <circle cx="30" cy="28" r="3" fill="#95D5B2"/>
            <circle cx="26" cy="37" r="3" fill="#95D5B2"/>
            <circle cx="32" cy="46" r="3" fill="#95D5B2"/>
            <circle cx="22" cy="55" r="3" fill="#95D5B2"/>
            <line x1="30" y1="28" x2="26" y2="37" stroke="#52B788" strokeWidth="0.8" strokeDasharray="2,2"/>
            <line x1="26" y1="37" x2="32" y2="46" stroke="#52B788" strokeWidth="0.8" strokeDasharray="2,2"/>
            <line x1="32" y1="46" x2="22" y2="55" stroke="#52B788" strokeWidth="0.8" strokeDasharray="2,2"/>
          </svg>
          <span className="text-green-700 text-xs font-semibold uppercase tracking-wider">DocuMind</span>
          {message.query_type && (
            <span className="text-xs text-gray-500 bg-gray-100 px-2 py-0.5 rounded-full border border-gray-200">
              {message.query_type}
            </span>
          )}
        </div>

        <div className="bg-white border border-gray-200 rounded-2xl rounded-tl-sm px-5 py-4 shadow-sm">
          <div className="md-answer text-gray-800 text-sm">
            <ReactMarkdown
              components={{
                h1: ({ children }) => <h1 className="font-bold text-xl mb-2" style={{ color: "#1B4332" }}>{children}</h1>,
                h2: ({ children }) => <h2 className="font-bold text-lg mb-2" style={{ color: "#1B4332" }}>{children}</h2>,
                h3: ({ children }) => <h3 className="font-bold text-base mb-1" style={{ color: "#1B4332" }}>{children}</h3>,
                strong: ({ children }) => <strong className="font-bold text-gray-800">{children}</strong>,
                em: ({ children }) => <em className="italic">{children}</em>,
                p: ({ children }) => <p className="mb-3">{children}</p>,
                ul: ({ children }) => <ul className="list-disc pl-5 mb-3">{children}</ul>,
                ol: ({ children }) => <ol className="list-decimal pl-5 mb-3">{children}</ol>,
                li: ({ children }) => <li className="mb-1">{children}</li>,
              }}
            >
              {message.content}
            </ReactMarkdown>
          </div>

          {Object.keys(groupedSources).length > 0 && message.content.includes("[Source") && (
            <div className="mt-4">
              <p className="text-gray-400 text-xs uppercase tracking-wider mb-2 font-semibold">
                Sources
              </p>
              {Object.entries(groupedSources).map(([filename, passages]) => (
                <DocumentSourceGroup key={filename} filename={filename} passages={passages} />
              ))}
            </div>
          )}

          {prevQuestion && (
            <ShareBar question={prevQuestion} answer={message.content} />
          )}
        </div>
      </div>
    </div>
  );
}

// ── Main app ───────────────────────────────────────────────────────────────────
export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [documents, setDocuments] = useState<DocRecord[]>([]);
  const [uploading, setUploading] = useState(false);
  const [uploadMsg, setUploadMsg] = useState("");
  const [listening, setListening] = useState(false);
  const [docSearch, setDocSearch] = useState("");
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);
  const [versionModal, setVersionModal] = useState<{ filename: string; currentVersion: number; file: File } | null>(null);
  const [tenantId, setTenantId] = useState("default");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [sidebarWidth, setSidebarWidth] = useState(272);
  const isDragging = useRef(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const lastAnswerRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const recognitionRef = useRef<any>(null);

  useEffect(() => {
    fetchDocuments();

    // Setup Web Speech API
    const SpeechRecognition =
      (window as any).SpeechRecognition ||
      (window as any).webkitSpeechRecognition;

    if (SpeechRecognition) {
      const recognition = new SpeechRecognition();
      recognition.continuous = false;
      recognition.interimResults = false;
      recognition.lang = "en-US";
      recognition.onresult = (event: any) => {
        setInput(event.results[0][0].transcript);
        setListening(false);
      };
      recognition.onerror = () => setListening(false);
      recognition.onend = () => setListening(false);
      recognitionRef.current = recognition;
    }
  }, []);

  // Refocus textarea when loading transitions from true to false
  useEffect(() => {
    if (!loading) {
      textareaRef.current?.focus();
    }
  }, [loading]);

  // #6 — Scroll to top of latest answer, not the bottom of the page
  useEffect(() => {
    if (!loading && messages.length > 0 && messages[messages.length - 1].role === "assistant") {
      setTimeout(() => {
        lastAnswerRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 100);
    } else {
      bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, loading]);

  // Sidebar drag-to-resize
  useEffect(() => {
    const onMouseMove = (e: MouseEvent) => {
      if (!isDragging.current) return;
      const newWidth = Math.min(400, Math.max(180, e.clientX));
      setSidebarWidth(newWidth);
    };
    const onMouseUp = () => {
      if (isDragging.current) {
        isDragging.current = false;
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
      }
    };
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
    return () => {
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
    };
  }, []);

  const startDrag = () => {
    isDragging.current = true;
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
  };

  const fetchDocuments = async () => {
    try {
      const res = await axios.get(`${API}/documents`, {
        headers: { "X-Tenant-ID": tenantId },
      });
      setDocuments(res.data.documents);
    } catch {
      console.error("Could not fetch documents");
    }
  };

  const toggleVoice = () => {
    if (!recognitionRef.current) return;
    if (listening) {
      recognitionRef.current.stop();
      setListening(false);
    } else {
      recognitionRef.current.start();
      setListening(true);
    }
  };

  // Split a raw input into distinct questions when multi-question signals are present.
  const splitQuestions = (text: string): string[] => {
    const trimmed = text.trim();

    // Signals that suggest more than one question
    const hasMultipleQuestionMarks = (trimmed.match(/\?/g) ?? []).length > 1;
    const hasConjunction = /\b(also|as well as|additionally)\b/i.test(trimmed);

    // Split on "and" only when both sides look like questions (contain a "?")
    const andParts = trimmed.split(/\s+and\s+/i);
    const andSplitYieldsQuestions =
      andParts.length > 1 && andParts.every((p) => p.includes("?"));

    if (!hasMultipleQuestionMarks && !hasConjunction && !andSplitYieldsQuestions) {
      return [trimmed];
    }

    // Prefer splitting on sentence boundaries (? followed by whitespace or end)
    const sentenceSplit = trimmed
      .split(/(?<=\?)\s+/)
      .map((s) => s.trim())
      .filter(Boolean);

    if (sentenceSplit.length > 1) return sentenceSplit;

    // Fallback: split on conjunction keywords
    const conjunctionSplit = trimmed
      .split(/\s*\b(also|as well as|additionally)\b\s*/i)
      .map((s) => s.trim())
      .filter((s) => s && !/^(also|as well as|additionally)$/i.test(s));

    if (conjunctionSplit.length > 1) return conjunctionSplit;

    return [trimmed];
  };

  const sendQuery = async (question: string) => {
    if (!question.trim() || loading) return;

    const questions = splitQuestions(question);

    const userMessage: Message = { role: "user", content: question };
    const baseMessages = [...messages, userMessage];
    setMessages(baseMessages);
    setInput("");
    setLoading(true);

    // Build history from everything before the user's new message
    const history = messages.map((m) => ({ role: m.role, content: m.content }));

    try {
      // Send each question sequentially; accumulate assistant replies after the user bubble
      let accumulated = baseMessages;
      for (const q of questions) {
        const res = await axios.post(`${API}/query`, { question: q, history }, {
          headers: { "X-Tenant-ID": tenantId },
        });
        console.log("Sources received:", res.data.sources);
        const assistantMessage: Message = {
          role: "assistant",
          content: res.data.answer,
          sources: res.data.sources,
          query_type: res.data.query_type,
        };
        accumulated = [...accumulated, assistantMessage];
        setMessages([...accumulated]);
        // Add each reply to history so later questions have context
        history.push({ role: "user", content: q });
        history.push({ role: "assistant", content: res.data.answer });
      }
    } catch {
      setMessages([
        ...baseMessages,
        { role: "assistant", content: "Something went wrong. Please try again." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const uploadFile = async (file: File, forceNewVersion: boolean) => {
    setUploading(true);
    setUploadMsg("");
    const formData = new FormData();
    formData.append("file", file);
    formData.append("force_new_version", forceNewVersion ? "true" : "false");
    try {
      const res = await axios.post(`${API}/upload`, formData, {
        headers: { "X-Tenant-ID": tenantId },
        timeout: 120000,
      });
      setUploadMsg(`✅ ${res.data.message}`);
      fetchDocuments();
    } catch (err: any) {
      // 409 = duplicate detected — show version modal
      if (err.response?.status === 409 && err.response?.data?.status === "duplicate") {
        const versions = err.response.data.existing_versions || [];
        const maxVer = versions.reduce((m: number, v: any) => Math.max(m, v.version || 1), 1);
        setVersionModal({ filename: file.name, currentVersion: maxVer, file });
        setUploading(false);
        return;
      }
      console.error("Upload error:", err);
      setUploadMsg(`❌ ${err.response?.data?.detail || err.message || "Upload failed"}`);
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Client-side validation
    const allowed = [".pdf", ".docx", ".txt"];
    const ext = file.name.toLowerCase().slice(file.name.lastIndexOf("."));
    if (!allowed.includes(ext)) {
      setUploadMsg("❌ Unsupported file type. Please upload a PDF, Word (.docx), or text (.txt) file.");
      if (fileInputRef.current) fileInputRef.current.value = "";
      return;
    }
    if (file.size > 20 * 1024 * 1024) {
      setUploadMsg("❌ File too large. Maximum allowed size is 20 MB.");
      if (fileInputRef.current) fileInputRef.current.value = "";
      return;
    }

    await uploadFile(file, false);
  };

  const handleDelete = async (filename: string) => {
    try {
      await axios.delete(`${API}/documents/${encodeURIComponent(filename)}`, {
        headers: { "X-Tenant-ID": tenantId },
      });
      setConfirmDelete(null);
      fetchDocuments();
    } catch {
      console.error("Delete failed");
    }
  };

  const getUserQuestion = (index: number): string | undefined => {
    if (index === 0) return undefined;
    const prev = messages[index - 1];
    return prev?.role === "user" ? prev.content : undefined;
  };

  // #3 — Filter documents by search query
  const filteredDocs = docSearch.trim()
    ? documents.filter((d) => d.filename.toLowerCase().includes(docSearch.toLowerCase()))
    : documents;

  return (
    <div className="flex h-screen bg-gray-50 text-gray-900 overflow-hidden">

      {/* ── Sidebar ── */}
      <aside
        className="shrink-0 bg-green-800 flex flex-col relative"
        style={{
          width: sidebarOpen ? `${sidebarWidth}px` : '0px',
          minWidth: sidebarOpen ? `${sidebarWidth}px` : '0px',
          overflow: 'hidden',
          transition: isDragging.current ? 'none' : 'width 0.25s ease, min-width 0.25s ease',
        }}
      >
        {/* Sidebar content wrapper — fixed width so content doesn't reflow during collapse */}
        <div className="flex flex-col h-full" style={{ width: `${sidebarWidth}px` }}>

        {/* Logo */}
        <div className="px-5 py-5 border-b border-green-700">
          <div className="flex items-center gap-3">
            <div style={{ width: '36px', height: '36px' }}>
              <svg viewBox="0 0 56 70" width="36" height="36" xmlns="http://www.w3.org/2000/svg">
                <rect x="0" y="0" width="56" height="70" rx="5" fill="#1B4332"/>
                <polygon points="38,0 56,18 38,18" fill="#2D6A4F"/>
                <polyline points="38,0 56,18 38,18 38,0" fill="none" stroke="#52B788" strokeWidth="0.8"/>
                <line x1="8" y1="28" x2="30" y2="28" stroke="#52B788" strokeWidth="1.3" strokeLinecap="round"/>
                <line x1="8" y1="37" x2="26" y2="37" stroke="#52B788" strokeWidth="1.3" strokeLinecap="round"/>
                <line x1="8" y1="46" x2="32" y2="46" stroke="#52B788" strokeWidth="1.3" strokeLinecap="round"/>
                <line x1="8" y1="55" x2="22" y2="55" stroke="#52B788" strokeWidth="1.3" strokeLinecap="round"/>
                <circle cx="30" cy="28" r="3" fill="#95D5B2"/>
                <circle cx="26" cy="37" r="3" fill="#95D5B2"/>
                <circle cx="32" cy="46" r="3" fill="#95D5B2"/>
                <circle cx="22" cy="55" r="3" fill="#95D5B2"/>
                <line x1="30" y1="28" x2="26" y2="37" stroke="#52B788" strokeWidth="0.8" strokeDasharray="2,2"/>
                <line x1="26" y1="37" x2="32" y2="46" stroke="#52B788" strokeWidth="0.8" strokeDasharray="2,2"/>
                <line x1="32" y1="46" x2="22" y2="55" stroke="#52B788" strokeWidth="0.8" strokeDasharray="2,2"/>
              </svg>
            </div>
            <div>
              <h1 style={{ fontSize: '22px', lineHeight: 1.1 }} className="font-bold">
                <span className="text-white">Docu</span>
                <span style={{ color: '#52B788' }}>Mind</span>
              </h1>
              <p style={{ fontSize: '11px', color: '#95D5B2' }}>Building Intelligence Platform</p>
            </div>
          </div>
        </div>

        {/* Document corpus */}
        <div className="px-4 py-4 border-b border-green-700">
          <p className="text-green-300 text-xs font-semibold uppercase tracking-wider mb-2">
            Document Corpus
          </p>

          {/* #3 — Search */}
          <input
            type="text"
            value={docSearch}
            onChange={(e) => setDocSearch(e.target.value)}
            placeholder="🔍 Search documents..."
            className="w-full bg-green-700 text-green-100 placeholder-green-400 text-xs px-3 py-1.5 rounded-lg outline-none focus:ring-1 focus:ring-green-400 mb-3"
          />

          <div className="space-y-2 max-h-52 overflow-y-auto">
            {filteredDocs.length === 0 && (
              <p className="text-green-400 text-xs">
                {docSearch ? "No documents match." : "No documents loaded yet."}
              </p>
            )}
            {filteredDocs.map((doc) => (
              <div key={doc.filename} className="flex items-start gap-2 py-1 group">
                <span className="text-green-300 mt-0.5 shrink-0">📄</span>
                <div className="flex-1 min-w-0">
                  <p className="text-green-100 text-xs leading-snug truncate">
                    {doc.filename.replace(".pdf", "").replace(/_/g, " ")}
                    {doc.version && doc.version > 1 && (
                      <span className="ml-1.5 bg-green-600 text-green-100 text-xs px-1.5 py-0.5 rounded-full">v{doc.version}</span>
                    )}
                  </p>
                  {/* #10 — Upload date */}
                  {doc.uploaded_at && (
                    <p className="text-green-400 text-xs mt-0.5">{formatDate(doc.uploaded_at)}</p>
                  )}
                </div>
                {/* #11 — Delete button */}
                <button
                  onClick={() => setConfirmDelete(doc.filename)}
                  title="Delete document"
                  className="shrink-0 text-green-500 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-opacity cursor-pointer text-xs mt-0.5"
                >
                  🗑️
                </button>
              </div>
            ))}
          </div>
        </div>

        {/* Upload */}
        <div className="px-4 py-4 border-b border-green-700">
          <p className="text-green-300 text-xs font-semibold uppercase tracking-wider mb-3">
            Upload Document
          </p>
          <input ref={fileInputRef} type="file" accept=".pdf,.docx,.txt" onChange={handleUpload} className="hidden" />
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={uploading}
            className="w-full bg-white hover:bg-green-50 disabled:opacity-50 text-green-800 font-semibold text-xs py-2 px-3 rounded-lg transition-colors flex items-center justify-center gap-2 cursor-pointer"
          >
            {uploading ? "Ingesting..." : "📤 Upload Document"}
          </button>
          <p className="text-green-500 text-xs mt-1 text-center">PDF, Word, or TXT · Max 20 MB</p>
          {uploadMsg && (
            <p className="text-xs mt-2 text-green-200 leading-snug">{uploadMsg}</p>
          )}
        </div>

        {/* Example queries */}
        <div className="px-4 py-4 flex-1 overflow-y-auto">
          <p className="text-green-300 text-xs font-semibold uppercase tracking-wider mb-3">
            Example Queries
          </p>
          <div className="space-y-2">
            {EXAMPLE_QUERIES.map((q, i) => (
              <button
                key={i}
                onClick={() => sendQuery(q)}
                disabled={loading}
                title={loading ? "Please wait — a question is being answered" : ""}
                className="w-full text-left text-xs text-green-100 hover:text-white bg-green-700 hover:bg-green-600 disabled:opacity-40 disabled:cursor-not-allowed px-3 py-2 rounded-lg transition-colors leading-snug cursor-pointer"
              >
                {q}
              </button>
            ))}
          </div>
        </div>

        {/* Workspace ID */}
        <div className="px-4 py-3 border-t border-green-700">
          <p className="text-green-300 text-xs font-semibold uppercase tracking-wider mb-2">
            Workspace ID
          </p>
          <input
            type="text"
            value={tenantId}
            onChange={(e) => setTenantId(e.target.value.trim() || "default")}
            placeholder="default"
            className="w-full bg-green-700 text-green-100 placeholder-green-400 text-xs px-3 py-1.5 rounded-lg outline-none focus:ring-1 focus:ring-green-400"
          />
          <p className="text-green-500 text-xs mt-1">
            Isolates your documents and queries
          </p>
        </div>

        {/* Clear chat */}
        <div className="px-4 py-3 border-t border-green-700">
          <button
            onClick={() => setMessages([])}
            className="w-full text-xs text-green-300 hover:text-red-300 transition-colors py-1 cursor-pointer"
          >
            🗑️ Clear Chat
          </button>
        </div>
        </div>

        {/* Drag handle */}
        {sidebarOpen && (
          <div
            onMouseDown={startDrag}
            className="absolute top-0 right-0 h-full group"
            style={{ width: '4px', cursor: 'col-resize', zIndex: 10 }}
          >
            <div className="w-full h-full opacity-0 group-hover:opacity-100 transition-opacity" style={{ backgroundColor: '#52B788' }} />
          </div>
        )}
      </aside>

      {/* ── Sidebar toggle button — sits between sidebar and main content ── */}
      <button
        onClick={() => setSidebarOpen(!sidebarOpen)}
        className="shrink-0 bg-green-800 hover:bg-green-700 text-green-300 hover:text-white cursor-pointer flex items-center justify-center"
        style={{
          width: '20px',
          height: '100%',
          borderLeft: sidebarOpen ? '1px solid #15803d' : 'none',
          borderRight: '1px solid #e5e7eb',
          transition: 'background-color 0.15s ease',
        }}
        title={sidebarOpen ? 'Collapse sidebar' : 'Expand sidebar'}
      >
        <svg width="12" height="12" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
          {sidebarOpen
            ? <polyline points="9,2 4,7 9,12" />
            : <polyline points="5,2 10,7 5,12" />
          }
        </svg>
      </button>

      {/* ── Main chat area ── */}
      <main className="flex-1 flex flex-col overflow-hidden bg-white">

        {/* Top bar */}
        <div className="px-6 py-4 border-b border-gray-200 flex items-center justify-between shrink-0 bg-white">
          <span className="text-gray-700 font-semibold" style={{ fontSize: '14px' }}>DocuMind</span>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
            <span className="text-green-600 text-xs font-medium">Live</span>
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-6 py-6 bg-gray-50">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center h-full text-center">
              <h2 style={{ fontSize: '40px', lineHeight: 1.1 }} className="font-bold mb-2">
                <span style={{ color: '#1B4332' }}>Docu</span>
                <span style={{ color: '#52B788' }}>Mind</span>
              </h2>
              <p className="text-gray-400 mb-4" style={{ fontSize: '14px' }}>Building Intelligence Platform</p>
              <p className="text-gray-400 text-sm max-w-md">
                Ask anything about roofing specs, building codes, glazing standards, or site safety. Every answer is cited from your document corpus.
              </p>
              <p className="text-green-800 text-xs max-w-md mt-3 bg-green-50 border border-green-200 rounded-lg px-4 py-2.5 font-medium">
                Tip: Refer to documents by their full name (e.g. "TPO product datasheet", "curtain wall installation manual") or keywords from their content — not by doc number.
              </p>
            </div>
          )}

          {messages.map((message, i) => {
            const isLastAnswer = i === messages.length - 1 && message.role === "assistant";
            return (
              <div key={i} ref={isLastAnswer ? lastAnswerRef : null}>
                <MessageBubble
                  message={message}
                  prevQuestion={getUserQuestion(i)}
                />
              </div>
            );
          })}

          {loading && (
            <div className="flex justify-start mb-6">
              <div className="max-w-3xl w-full">
                <div className="flex items-center gap-2 mb-2">
                  <svg viewBox="0 0 56 70" width="22" height="22" xmlns="http://www.w3.org/2000/svg">
                    <rect x="0" y="0" width="56" height="70" rx="5" fill="#1B4332"/>
                    <polygon points="38,0 56,18 38,18" fill="#2D6A4F"/>
                    <polyline points="38,0 56,18 38,18 38,0" fill="none" stroke="#52B788" strokeWidth="0.8"/>
                    <line x1="8" y1="28" x2="30" y2="28" stroke="#52B788" strokeWidth="1.3" strokeLinecap="round"/>
                    <line x1="8" y1="37" x2="26" y2="37" stroke="#52B788" strokeWidth="1.3" strokeLinecap="round"/>
                    <line x1="8" y1="46" x2="32" y2="46" stroke="#52B788" strokeWidth="1.3" strokeLinecap="round"/>
                    <line x1="8" y1="55" x2="22" y2="55" stroke="#52B788" strokeWidth="1.3" strokeLinecap="round"/>
                    <circle cx="30" cy="28" r="3" fill="#95D5B2"/>
                    <circle cx="26" cy="37" r="3" fill="#95D5B2"/>
                    <circle cx="32" cy="46" r="3" fill="#95D5B2"/>
                    <circle cx="22" cy="55" r="3" fill="#95D5B2"/>
                    <line x1="30" y1="28" x2="26" y2="37" stroke="#52B788" strokeWidth="0.8" strokeDasharray="2,2"/>
                    <line x1="26" y1="37" x2="32" y2="46" stroke="#52B788" strokeWidth="0.8" strokeDasharray="2,2"/>
                    <line x1="32" y1="46" x2="22" y2="55" stroke="#52B788" strokeWidth="0.8" strokeDasharray="2,2"/>
                  </svg>
                  <span className="text-green-700 text-xs font-semibold uppercase tracking-wider">DocuMind</span>
                </div>
                <div className="bg-white border border-gray-200 rounded-2xl rounded-tl-sm px-5 py-4 shadow-sm">
                  <div className="flex gap-1.5 items-center">
                    <div className="w-2 h-2 bg-green-500 rounded-full animate-bounce" style={{ animationDelay: "0ms" }}></div>
                    <div className="w-2 h-2 bg-green-500 rounded-full animate-bounce" style={{ animationDelay: "150ms" }}></div>
                    <div className="w-2 h-2 bg-green-500 rounded-full animate-bounce" style={{ animationDelay: "300ms" }}></div>
                    <span className="text-gray-400 text-xs ml-2">Searching documents...</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          <div ref={bottomRef} />
        </div>


        {/* Input bar */}
        <div className="px-6 py-4 border-t border-gray-200 bg-white shrink-0">
          <div className="flex gap-2 items-end bg-gray-50 border border-gray-300 focus-within:border-green-600 rounded-xl transition-colors px-3 py-2">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  sendQuery(input);
                }
              }}
              disabled={loading}
              placeholder={loading ? "Please wait..." : "Ask about roofing specs, building codes, glazing standards, or safety..."}
              rows={1}
              className="flex-1 bg-transparent text-gray-900 text-sm py-1 resize-none outline-none placeholder-gray-400 disabled:cursor-not-allowed"
            />

            {/* Mic button */}
            <button
              onClick={toggleVoice}
              disabled={loading}
              title={listening ? "Stop listening" : "Speak your question"}
              className={`p-1.5 rounded-lg transition-colors shrink-0 cursor-pointer ${
                listening
                  ? "text-red-500 bg-red-50 animate-pulse"
                  : "text-gray-400 hover:text-green-600 hover:bg-green-50"
              }`}
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm-1-9c0-.55.45-1 1-1s1 .45 1 1v6c0 .55-.45 1-1 1s-1-.45-1-1V5zm6 6c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/>
              </svg>
            </button>

            {/* Send button */}
            <button
              onClick={() => sendQuery(input)}
              disabled={loading || !input.trim()}
              className="bg-green-700 hover:bg-green-600 disabled:opacity-40 disabled:cursor-not-allowed text-white px-4 py-1.5 rounded-lg transition-colors shrink-0 text-sm font-medium cursor-pointer"
            >
              Send
            </button>
          </div>
          <p className="text-gray-400 text-xs mt-2 text-center">
            Press Enter to send · Shift+Enter for new line · 🎤 for voice
          </p>
        </div>
      </main>

      {/* ── #11 Delete confirmation modal ── */}
      {confirmDelete && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-2xl shadow-xl p-6 max-w-sm w-full mx-4">
            <h3 className="text-gray-900 font-semibold text-base mb-2">Delete Document</h3>
            <p className="text-gray-600 text-sm mb-5">
              Delete <strong>{confirmDelete.replace(".pdf", "").replace(/_/g, " ")}</strong>? This removes all its indexed content and cannot be undone.
            </p>
            <div className="flex gap-3">
              <button
                onClick={() => handleDelete(confirmDelete)}
                className="flex-1 bg-red-600 hover:bg-red-700 text-white text-sm font-semibold py-2 rounded-lg transition-colors cursor-pointer"
              >
                Yes, delete
              </button>
              <button
                onClick={() => setConfirmDelete(null)}
                className="flex-1 bg-gray-100 hover:bg-gray-200 text-gray-700 text-sm font-semibold py-2 rounded-lg transition-colors cursor-pointer"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Version conflict modal */}
      {versionModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-2xl shadow-xl p-6 max-w-md w-full mx-4">
            <h3 className="text-gray-900 font-semibold text-base mb-2">Document Already Exists</h3>
            <p className="text-gray-600 text-sm mb-5">
              A document named <strong>{versionModal.filename.replace(".pdf", "").replace(/_/g, " ")}</strong> already exists
              (currently on version <strong>{versionModal.currentVersion}</strong>). What would you like to do?
            </p>
            <div className="flex gap-3">
              <button
                onClick={() => {
                  const file = versionModal.file;
                  setVersionModal(null);
                  uploadFile(file, true);
                }}
                className="flex-1 bg-green-700 hover:bg-green-800 text-white text-sm font-semibold py-2 rounded-lg transition-colors cursor-pointer"
              >
                Save as New Version
              </button>
              <button
                onClick={() => {
                  setVersionModal(null);
                  if (fileInputRef.current) fileInputRef.current.value = "";
                }}
                className="flex-1 bg-gray-100 hover:bg-gray-200 text-gray-700 text-sm font-semibold py-2 rounded-lg transition-colors cursor-pointer"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
