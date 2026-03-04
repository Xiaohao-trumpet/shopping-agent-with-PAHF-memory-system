import { useEffect, useMemo, useState } from "react";
import {
  addMemory,
  fetchHealth,
  findSimilarMemory,
  fetchModels,
  fetchPromptScenes,
  listMemories,
  searchMemories,
  sendChatCompletion,
  updateMemory,
} from "./api";
import type { Conversation, MemoryItem, MemorySearchHit } from "./types";

const STORAGE_KEY = "servicebot_conversations_v1";
const MODEL_KEY = "servicebot_selected_model";
const USER_KEY = "servicebot_user_id";
const SCENE_KEY = "servicebot_scene";
const SYSTEM_KEY = "servicebot_system_prompt";

function nowIso(): string {
  return new Date().toISOString();
}

function createConversation(model: string, userId: string, scene: string, systemPrompt: string): Conversation {
  const id = crypto.randomUUID();
  return {
    id,
    title: "New Conversation",
    model,
    userId,
    scene,
    systemPrompt,
    messages: [],
    createdAt: nowIso(),
    updatedAt: nowIso(),
  };
}

export default function App() {
  const [health, setHealth] = useState("unknown");
  const [models, setModels] = useState<string[]>([]);
  const [scenes, setScenes] = useState<string[]>(["default", "it_helpdesk"]);

  const [userId, setUserId] = useState(localStorage.getItem(USER_KEY) ?? "demo_user");
  const [selectedModel, setSelectedModel] = useState(localStorage.getItem(MODEL_KEY) ?? "");
  const [selectedScene, setSelectedScene] = useState(localStorage.getItem(SCENE_KEY) ?? "default");
  const [systemPrompt, setSystemPrompt] = useState(localStorage.getItem(SYSTEM_KEY) ?? "");

  const [conversations, setConversations] = useState<Conversation[]>(() => {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    try {
      return JSON.parse(raw) as Conversation[];
    } catch {
      return [];
    }
  });
  const [activeConversationId, setActiveConversationId] = useState<string>("");
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);

  const [memories, setMemories] = useState<MemoryItem[]>([]);
  const [memoryText, setMemoryText] = useState("");
  const [memoryUpdateId, setMemoryUpdateId] = useState("");
  const [memoryUpdateText, setMemoryUpdateText] = useState("");
  const [memoryQuery, setMemoryQuery] = useState("");
  const [memoryHits, setMemoryHits] = useState<MemorySearchHit[]>([]);
  const [similarMemory, setSimilarMemory] = useState<MemoryItem | null>(null);

  const activeConversation = useMemo(
    () => conversations.find((c) => c.id === activeConversationId),
    [conversations, activeConversationId]
  );

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations));
  }, [conversations]);

  useEffect(() => {
    localStorage.setItem(USER_KEY, userId);
  }, [userId]);

  useEffect(() => {
    localStorage.setItem(MODEL_KEY, selectedModel);
  }, [selectedModel]);

  useEffect(() => {
    localStorage.setItem(SCENE_KEY, selectedScene);
  }, [selectedScene]);

  useEffect(() => {
    localStorage.setItem(SYSTEM_KEY, systemPrompt);
  }, [systemPrompt]);

  useEffect(() => {
    async function bootstrap() {
      try {
        const [healthRes, modelRes, sceneRes] = await Promise.all([
          fetchHealth(),
          fetchModels(),
          fetchPromptScenes().catch(() => ({ scenes: ["default", "it_helpdesk"], default_scene: "default" })),
        ]);
        setHealth(healthRes.status);
        setModels(modelRes.map((m) => m.id));
        setScenes(sceneRes.scenes || ["default", "it_helpdesk"]);
        if (!selectedModel && modelRes.length > 0) {
          setSelectedModel(modelRes[0].id);
        }
      } catch {
        setHealth("down");
      }
    }
    void bootstrap();
  }, [selectedModel]);

  useEffect(() => {
    void refreshMemories();
  }, [userId]);

  function ensureConversation(): Conversation {
    if (activeConversation) return activeConversation;
    const baseModel = selectedModel || "default-model";
    const conversation = createConversation(baseModel, userId, selectedScene, systemPrompt);
    setConversations((prev) => [conversation, ...prev]);
    setActiveConversationId(conversation.id);
    return conversation;
  }

  function updateConversation(conversationId: string, updater: (conv: Conversation) => Conversation) {
    setConversations((prev) => prev.map((c) => (c.id === conversationId ? updater(c) : c)));
  }

  async function refreshMemories() {
    try {
      const items = await listMemories(userId);
      setMemories(items);
    } catch {
      setMemories([]);
    }
  }

  async function sendMessage() {
    if (!input.trim() || sending) return;
    const conv = ensureConversation();
    const messageText = input.trim();
    setInput("");
    setSending(true);

    const userMessage = {
      id: crypto.randomUUID(),
      role: "user" as const,
      content: messageText,
      createdAt: nowIso(),
      trace: null,
    };

    updateConversation(conv.id, (current) => ({
      ...current,
      title: current.messages.length === 0 ? messageText.slice(0, 30) || "Conversation" : current.title,
      updatedAt: nowIso(),
      model: selectedModel || current.model,
      userId,
      scene: selectedScene,
      systemPrompt,
      messages: [...current.messages, userMessage],
    }));

    try {
      const source = conversations.find((c) => c.id === conv.id) ?? conv;
      const payloadMessages = [...source.messages, userMessage].map((m) => ({
        role: m.role,
        content: m.content,
      }));
      const response = await sendChatCompletion({
        model: selectedModel || source.model,
        userId,
        scene: selectedScene,
        systemPrompt,
        messages: payloadMessages,
      });

      const assistantMessage = {
        id: crypto.randomUUID(),
        role: "assistant" as const,
        content: response.content || "(empty response)",
        createdAt: nowIso(),
        trace: response.trace,
      };

      updateConversation(conv.id, (current) => ({
        ...current,
        updatedAt: nowIso(),
        messages: [...current.messages, assistantMessage],
      }));
      await refreshMemories();
    } catch (error) {
      const errText = error instanceof Error ? error.message : "Failed to send message";
      const assistantMessage = {
        id: crypto.randomUUID(),
        role: "assistant" as const,
        content: `Error: ${errText}`,
        createdAt: nowIso(),
        trace: null,
      };
      updateConversation(conv.id, (current) => ({
        ...current,
        updatedAt: nowIso(),
        messages: [...current.messages, assistantMessage],
      }));
    } finally {
      setSending(false);
    }
  }

  function onNewConversation() {
    const baseModel = selectedModel || models[0] || "default-model";
    const c = createConversation(baseModel, userId, selectedScene, systemPrompt);
    setConversations((prev) => [c, ...prev]);
    setActiveConversationId(c.id);
  }

  function onDeleteConversation(id: string) {
    setConversations((prev) => prev.filter((c) => c.id !== id));
    if (activeConversationId === id) {
      const remaining = conversations.filter((c) => c.id !== id);
      setActiveConversationId(remaining[0]?.id ?? "");
    }
  }

  async function onAddMemory() {
    if (!memoryText.trim()) return;
    await addMemory(userId, memoryText.trim());
    setMemoryText("");
    await refreshMemories();
  }

  async function onUpdateMemory() {
    const memoryId = Number(memoryUpdateId);
    if (!Number.isFinite(memoryId) || memoryId <= 0 || !memoryUpdateText.trim()) return;
    await updateMemory(memoryId, userId, memoryUpdateText.trim());
    setMemoryUpdateId("");
    setMemoryUpdateText("");
    await refreshMemories();
  }

  async function onSearchMemory() {
    if (!memoryQuery.trim()) {
      setMemoryHits([]);
      return;
    }
    const hits = await searchMemories(userId, memoryQuery.trim(), 5);
    setMemoryHits(hits);
  }

  async function onFindSimilar() {
    if (!memoryQuery.trim()) {
      setSimilarMemory(null);
      return;
    }
    const item = await findSimilarMemory(userId, memoryQuery.trim());
    setSimilarMemory(item);
  }

  const lastAssistantTrace =
    activeConversation?.messages
      .filter((m) => m.role === "assistant")
      .slice(-1)[0]?.trace ?? null;
  const lastRetrievedMemories = Array.isArray(lastAssistantTrace?.retrieved_memories)
    ? (lastAssistantTrace.retrieved_memories as Array<{ id?: number; text?: string; score?: number }>)
    : [];

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="sidebar-header">
          <h1>ServiceBot</h1>
          <button onClick={onNewConversation}>+ New</button>
        </div>
        <div className="status-row">
          <span className={`status-dot ${health === "ok" ? "ok" : "bad"}`} />
          <span>Backend: {health}</span>
        </div>
        <div className="conversation-list">
          {conversations.map((conv) => (
            <div
              className={`conversation-item ${activeConversationId === conv.id ? "active" : ""}`}
              key={conv.id}
            >
              <button className="conversation-select" onClick={() => setActiveConversationId(conv.id)}>
                {conv.title}
              </button>
              <button className="danger-link" onClick={() => onDeleteConversation(conv.id)}>
                x
              </button>
            </div>
          ))}
          {conversations.length === 0 ? <p className="muted">No conversations yet.</p> : null}
        </div>
      </aside>

      <main className="chat-area">
        <header className="topbar">
          <label>
            User ID
            <input value={userId} onChange={(e) => setUserId(e.target.value)} />
          </label>
          <label>
            Model
            <select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)}>
              {models.map((m) => (
                <option value={m} key={m}>
                  {m}
                </option>
              ))}
            </select>
          </label>
          <label>
            Scene
            <select value={selectedScene} onChange={(e) => setSelectedScene(e.target.value)}>
              {scenes.map((scene) => (
                <option value={scene} key={scene}>
                  {scene}
                </option>
              ))}
            </select>
          </label>
        </header>

        <section className="system-prompt">
          <label>
            System Prompt (optional)
            <textarea
              value={systemPrompt}
              onChange={(e) => setSystemPrompt(e.target.value)}
              placeholder="Provide optional system instructions..."
            />
          </label>
        </section>

        <section className="messages">
          {(activeConversation?.messages ?? []).map((m) => (
            <div key={m.id} className={`message ${m.role}`}>
              <div className="bubble">
                <p>{m.content}</p>
              </div>
            </div>
          ))}
          {!activeConversation || activeConversation.messages.length === 0 ? (
            <p className="muted">Start a conversation.</p>
          ) : null}
        </section>

        <section className="composer">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type your message..."
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                void sendMessage();
              }
            }}
          />
          <button disabled={sending} onClick={() => void sendMessage()}>
            {sending ? "Sending..." : "Send"}
          </button>
        </section>
      </main>

      <aside className="right-panel">
        <section className="panel-card">
          <h3>PAHF Memory</h3>
          <div className="memory-form">
            <textarea
              value={memoryText}
              onChange={(e) => setMemoryText(e.target.value)}
              placeholder="Add memory text..."
            />
            <button onClick={() => void onAddMemory()}>Add Memory</button>
          </div>

          <div className="memory-form">
            <input
              value={memoryUpdateId}
              onChange={(e) => setMemoryUpdateId(e.target.value)}
              placeholder="Memory ID to update"
            />
            <textarea
              value={memoryUpdateText}
              onChange={(e) => setMemoryUpdateText(e.target.value)}
              placeholder="Updated memory text..."
            />
            <button onClick={() => void onUpdateMemory()}>Update Memory</button>
          </div>

          <div className="memory-search">
            <input
              value={memoryQuery}
              onChange={(e) => setMemoryQuery(e.target.value)}
              placeholder="Search memory..."
            />
            <button onClick={() => void onSearchMemory()}>Search</button>
            <button onClick={() => void onFindSimilar()}>Find Similar</button>
          </div>

          {memoryHits.length > 0 ? (
            <div className="memory-hits">
              <h4>Search Results</h4>
              {memoryHits.map((hit) => (
                <div key={hit.memory.id} className="memory-item">
                  <small>score={hit.score.toFixed(3)}</small>
                  <p>{hit.memory.text}</p>
                </div>
              ))}
            </div>
          ) : null}

          {similarMemory ? (
            <div className="memory-hits">
              <h4>Most Similar</h4>
              <div className="memory-item">
                <small>id={similarMemory.id}</small>
                <p>{similarMemory.text}</p>
              </div>
            </div>
          ) : null}

          {lastRetrievedMemories.length > 0 ? (
            <div className="memory-hits">
              <h4>Retrieved (Last Reply)</h4>
              {lastRetrievedMemories.map((item, idx) => (
                <div key={`${item.id ?? idx}`} className="memory-item">
                  <small>id={item.id ?? "n/a"} score={typeof item.score === "number" ? item.score.toFixed(3) : "n/a"}</small>
                  <p>{item.text ?? ""}</p>
                </div>
              ))}
            </div>
          ) : null}

          <div className="memory-list">
            <h4>Stored Memories</h4>
            {memories.map((item) => (
              <div key={item.id} className="memory-item">
                <small>id={item.id}</small>
                <p>{item.text}</p>
              </div>
            ))}
            {memories.length === 0 ? <p className="muted">No memories yet.</p> : null}
          </div>
        </section>

        <section className="panel-card">
          <h3>Tools / Trace</h3>
          {lastAssistantTrace ? (
            <pre className="trace-box">{JSON.stringify(lastAssistantTrace, null, 2)}</pre>
          ) : (
            <p className="muted">No trace metadata for last response.</p>
          )}
        </section>
      </aside>
    </div>
  );
}
