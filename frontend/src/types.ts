export type Role = "user" | "assistant";

export interface ModelInfo {
  id: string;
}

export interface ChatMessage {
  id: string;
  role: Role;
  content: string;
  createdAt: string;
  trace?: Record<string, unknown> | null;
}

export interface Conversation {
  id: string;
  title: string;
  model: string;
  userId: string;
  scene: string;
  systemPrompt: string;
  messages: ChatMessage[];
  createdAt: string;
  updatedAt: string;
}

export interface MemoryItem {
  id: number;
  person_id: string;
  text: string;
}

export interface MemorySearchHit {
  memory: MemoryItem;
  score: number;
}
