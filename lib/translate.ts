// Translation between the three provider wires, and a normalised answer.
// Every upstream answer is first reduced to {text, inputTokens, outputTokens, stop}
// (the provider's text whole, its usage counts exactly as it reported them, and
// its stop reason reduced to one of four families), then rendered in the wire
// the client used. Buffering a streamed upstream answer before rendering is
// deliberate: the simulated providers only report usage in their final frame,
// and the client checks usage and stop reason against the provider's own
// record, so the downstream stream must carry those final values.

import { isRecord, type Json } from './providers.ts';
import { sseEvent } from './wire.ts';

export type StopFamily = 'stop' | 'length' | 'tool_calls' | 'content_filter';

export interface Normalised {
  text: string;
  inputTokens: number;
  outputTokens: number;
  stop: StopFamily;
}

function num(v: unknown): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : 0;
}

function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('');
}

export function stopFamilyOf(v: unknown): StopFamily | null {
  if (v === 'stop' || v === 'end_turn' || v === 'STOP') return 'stop';
  if (v === 'length' || v === 'max_tokens' || v === 'MAX_TOKENS') return 'length';
  if (v === 'tool_calls' || v === 'function_call' || v === 'tool_use') return 'tool_calls';
  if (v === 'content_filter' || v === 'refusal' || v === 'SAFETY') return 'content_filter';
  return null;
}

function openAIStop(s: StopFamily): string {
  return s === 'stop' ? 'stop' : s === 'length' ? 'length' : s === 'tool_calls' ? 'tool_calls' : 'content_filter';
}
function anthropicStop(s: StopFamily): string {
  return s === 'stop' ? 'end_turn' : s === 'length' ? 'max_tokens' : s === 'tool_calls' ? 'tool_use' : 'refusal';
}

/** Split text into deterministic 13-char deltas, as the providers do. */
export function split13(text: string): string[] {
  const out: string[] = [];
  for (let at = 0; at < text.length; at += 13) out.push(text.slice(at, at + 13));
  return out.length === 0 ? [''] : out;
}

/** Parse SSE text into frames {event, data}. */
export function parseSse(text: string): { event: string | null; data: string }[] {
  const frames: { event: string | null; data: string }[] = [];
  const norm = text.replace(/\r\n/g, '\n');
  for (const frame of norm.split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice(6).trim();
      else if (line.startsWith('data:')) data.push(line.slice(5).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function parseJsonSafe(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

/** An OpenAI JSON or SSE answer -> normalised, or null. */
export async function normaliseOpenAI(res: Response, stream: boolean): Promise<Normalised | null> {
  const text = await res.text();
  if (!stream) {
    const body = parseJsonSafe(text);
    if (!isRecord(body)) return null;
    const choices = Array.isArray(body.choices) ? body.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) return null;
    const message = isRecord(choice.message) ? choice.message : {};
    const content = typeof message.content === 'string' ? message.content : textOf(message.content);
    const usage = isRecord(body.usage) ? body.usage : {};
    const stop = stopFamilyOf(choice.finish_reason);
    if (stop === null) return null;
    return { text: content, inputTokens: num(usage.prompt_tokens), outputTokens: num(usage.completion_tokens), stop };
  }
  let outText = '';
  let input = 0;
  let output = 0;
  let stop: StopFamily | null = null;
  for (const frame of parseSse(text)) {
    if (frame.data.trim() === '[DONE]') continue;
    const chunk = parseJsonSafe(frame.data);
    if (!isRecord(chunk)) continue;
    if (isRecord(chunk.usage)) {
      input = num(chunk.usage.prompt_tokens);
      output = num(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') outText += delta.content;
    if (typeof choice.finish_reason === 'string') stop = stopFamilyOf(choice.finish_reason);
  }
  if (stop === null) return null;
  return { text: outText, inputTokens: input, outputTokens: output, stop };
}

/** An Anthropic JSON or SSE answer -> normalised, or null. */
export async function normaliseAnthropic(res: Response, stream: boolean): Promise<Normalised | null> {
  const text = await res.text();
  if (!stream) {
    const body = parseJsonSafe(text);
    if (!isRecord(body) || !Array.isArray(body.content)) return null;
    const usage = isRecord(body.usage) ? body.usage : {};
    const stop = stopFamilyOf(body.stop_reason);
    if (stop === null) return null;
    return { text: textOf(body.content), inputTokens: num(usage.input_tokens), outputTokens: num(usage.output_tokens), stop };
  }
  let outText = '';
  let input = 0;
  let output = 0;
  let stop: StopFamily | null = null;
  for (const frame of parseSse(text)) {
    const data = parseJsonSafe(frame.data);
    if (!isRecord(data)) continue;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      input = num(data.message.usage.input_tokens);
    }
    if (frame.event === 'content_block_delta' && isRecord(data.delta) && data.delta.type === 'text_delta' && typeof data.delta.text === 'string') {
      outText += data.delta.text;
    }
    if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stop = stopFamilyOf(data.delta.stop_reason);
      if (isRecord(data.usage)) output = num(data.usage.output_tokens);
    }
  }
  if (stop === null) return null;
  return { text: outText, inputTokens: input, outputTokens: output, stop };
}

/** A Gemini JSON or SSE answer -> normalised, or null. */
export async function normaliseGemini(res: Response, stream: boolean): Promise<Normalised | null> {
  const text = await res.text();
  const bodies: unknown[] = [];
  if (!stream) {
    const body = parseJsonSafe(text);
    if (body !== null) bodies.push(body);
  } else {
    for (const frame of parseSse(text)) {
      const body = parseJsonSafe(frame.data);
      if (body !== null) bodies.push(body);
    }
  }
  let outText = '';
  let input = 0;
  let output = 0;
  let stop: StopFamily | null = null;
  for (const body of bodies) {
    if (!isRecord(body)) continue;
    const candidates = Array.isArray(body.candidates) ? body.candidates : [];
    const cand = isRecord(candidates[0]) ? candidates[0] : null;
    if (cand !== null) {
      const content = isRecord(cand.content) ? cand.content : {};
      const parts = Array.isArray(content.parts) ? content.parts : [];
      for (const part of parts) {
        if (isRecord(part) && typeof part.text === 'string') outText += part.text;
      }
      if (typeof cand.finishReason === 'string') stop = stopFamilyOf(cand.finishReason);
    }
    if (isRecord(body.usageMetadata)) {
      input = num(body.usageMetadata.promptTokenCount);
      output = num(body.usageMetadata.candidatesTokenCount);
    }
  }
  if (stop === null) return null;
  return { text: outText, inputTokens: input, outputTokens: output, stop };
}

/** Render a normalised answer as an OpenAI JSON completion. */
export function renderOpenAI(ans: Normalised, model: string): Json {
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: 1767225600,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: ans.text }, finish_reason: openAIStop(ans.stop), logprobs: null }],
    usage: { prompt_tokens: ans.inputTokens, completion_tokens: ans.outputTokens, total_tokens: ans.inputTokens + ans.outputTokens },
  };
}

/** Render a normalised answer as an OpenAI SSE stream, with a usage chunk. */
export function renderOpenAIStream(ans: Normalised, model: string): string {
  const frames: string[] = [];
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-sim',
    object: 'chat.completion.chunk',
    created: 1767225600,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  frames.push(sseEvent(null, chunk({ role: 'assistant', content: '' }, null)));
  for (const piece of split13(ans.text)) frames.push(sseEvent(null, chunk({ content: piece }, null)));
  frames.push(sseEvent(null, chunk({}, openAIStop(ans.stop))));
  frames.push(
    sseEvent(null, {
      id: 'chatcmpl-sim',
      object: 'chat.completion.chunk',
      created: 1767225600,
      model,
      choices: [],
      usage: { prompt_tokens: ans.inputTokens, completion_tokens: ans.outputTokens, total_tokens: ans.inputTokens + ans.outputTokens },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** Render a normalised answer as an Anthropic JSON message. */
export function renderAnthropic(ans: Normalised, model: string): Json {
  return {
    id: 'msg_sim',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: ans.text }],
    stop_reason: anthropicStop(ans.stop),
    stop_sequence: null,
    usage: { input_tokens: ans.inputTokens, output_tokens: ans.outputTokens },
  };
}

/** Render a normalised answer as an Anthropic SSE stream. */
export function renderAnthropicStream(ans: Normalised, model: string): string {
  const frames: string[] = [];
  frames.push(
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_sim',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: ans.inputTokens, output_tokens: 0 },
      },
    }),
  );
  frames.push(sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }));
  for (const piece of split13(ans.text)) {
    frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
  }
  frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }));
  frames.push(
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: anthropicStop(ans.stop), stop_sequence: null },
      usage: { output_tokens: ans.outputTokens },
    }),
  );
  frames.push(sseEvent('message_stop', { type: 'message_stop' }));
  return frames.join('');
}

function systemTextOfOpenAI(body: Json): string {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const parts: string[] = [];
  for (const m of messages) {
    if (isRecord(m) && m.role === 'system') {
      const t = textOf(m.content);
      if (t !== '') parts.push(t);
    }
  }
  return parts.join('\n');
}

/** An OpenAI Chat Completions request as an Anthropic Messages request. */
export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const messages: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') continue;
    if (raw.role === 'assistant') {
      const blocks: Json[] = [];
      const t = textOf(raw.content);
      if (t !== '') blocks.push({ type: 'text', text: t });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        let input: Json = {};
        if (typeof call.function.arguments === 'string') {
          const parsed = parseJsonSafe(call.function.arguments);
          if (isRecord(parsed)) input = parsed;
        }
        blocks.push({ type: 'tool_use', id: typeof call.id === 'string' ? call.id : 'toolu_unknown', name: typeof call.function.name === 'string' ? call.function.name : '', input });
      }
      if (blocks.length > 0) messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({ role: 'user', content: [{ type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: typeof raw.content === 'string' ? raw.content : textOf(raw.content) }] });
      continue;
    }
    // user
    if (typeof raw.content === 'string') {
      messages.push({ role: 'user', content: raw.content });
      continue;
    }
    if (Array.isArray(raw.content)) {
      const blocks: Json[] = [];
      for (const part of raw.content) {
        if (!isRecord(part)) continue;
        if (part.type === 'text' && typeof part.text === 'string') blocks.push({ type: 'text', text: part.text });
        else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
          const url = part.image_url.url;
          const m = /^data:([^;]+);base64,(.*)$/s.exec(url);
          if (m !== null) blocks.push({ type: 'image', source: { type: 'base64', media_type: m[1], data: m[2] } });
          else blocks.push({ type: 'image', source: { type: 'url', url } });
        }
      }
      if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
    }
  }
  const out: Json = { model, messages };
  const system = systemTextOfOpenAI(body);
  if (system !== '') out.system = system;
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') out.max_tokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop.filter((s) => typeof s === 'string');
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [{ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
    });
    if (tools.length > 0) out.tools = tools;
  }
  if (body.tool_choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (body.tool_choice === 'required') out.tool_choice = { type: 'any' };
  else if (isRecord(body.tool_choice) && isRecord(body.tool_choice.function) && typeof body.tool_choice.function.name === 'string') out.tool_choice = { type: 'tool', name: body.tool_choice.function.name };
  if (body.stream === true) out.stream = true;
  return out;
}

interface GeminiPart {
  text?: string;
  inlineData?: { mimeType: string; data: string };
  fileData?: { fileUri: string };
}

/** A client request (OpenAI or Anthropic) as a Gemini generateContent body. */
export function toGeminiRequest(clientApi: 'openai' | 'anthropic', body: Json, model: string, jsonMode: boolean): Json {
  let system = '';
  const contents: Json[] = [];
  const tools: Json[] = [];
  let maxTokens: number | undefined;
  let temperature: number | undefined;
  let topP: number | undefined;
  let stops: string[] = [];

  const addUserParts = (parts: GeminiPart[]): void => {
    if (parts.length > 0) contents.push({ role: 'user', parts });
  };

  if (clientApi === 'openai') {
    system = systemTextOfOpenAI(body);
    const messages = Array.isArray(body.messages) ? body.messages : [];
    for (const raw of messages) {
      if (!isRecord(raw)) continue;
      if (raw.role === 'system') continue;
      const role = raw.role === 'assistant' ? 'model' : 'user';
      const parts: GeminiPart[] = [];
      if (typeof raw.content === 'string') {
        if (raw.content !== '') parts.push({ text: raw.content });
      } else if (Array.isArray(raw.content)) {
        for (const part of raw.content) {
          if (!isRecord(part)) continue;
          if (part.type === 'text' && typeof part.text === 'string') parts.push({ text: part.text });
          else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
            const url = part.image_url.url;
            const m = /^data:([^;]+);base64,(.*)$/s.exec(url);
            if (m !== null) parts.push({ inlineData: { mimeType: m[1], data: m[2] } });
            else parts.push({ fileData: { fileUri: url } });
          }
        }
      }
      if (parts.length > 0) contents.push({ role, parts });
    }
    if (Array.isArray(body.tools)) {
      for (const raw of body.tools) {
        if (isRecord(raw) && isRecord(raw.function) && typeof raw.function.name === 'string') {
          tools.push({ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } });
        }
      }
    }
    if (typeof body.max_tokens === 'number') maxTokens = body.max_tokens;
    else if (typeof body.max_completion_tokens === 'number') maxTokens = body.max_completion_tokens;
    if (typeof body.temperature === 'number') temperature = body.temperature;
    if (typeof body.top_p === 'number') topP = body.top_p;
    if (Array.isArray(body.stop)) stops = body.stop.filter((s) => typeof s === 'string');
  } else {
    system = textOf(body.system);
    const messages = Array.isArray(body.messages) ? body.messages : [];
    for (const raw of messages) {
      if (!isRecord(raw)) continue;
      const role = raw.role === 'assistant' ? 'model' : 'user';
      const parts: GeminiPart[] = [];
      if (typeof raw.content === 'string') {
        if (raw.content !== '') parts.push({ text: raw.content });
      } else if (Array.isArray(raw.content)) {
        for (const block of raw.content) {
          if (!isRecord(block)) continue;
          if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
          else if (block.type === 'image' && isRecord(block.source)) {
            if (block.source.type === 'base64' && typeof block.source.data === 'string') {
              parts.push({ inlineData: { mimeType: typeof block.source.media_type === 'string' ? block.source.media_type : 'image/png', data: block.source.data } });
            } else if (block.source.type === 'url' && typeof block.source.url === 'string') {
              parts.push({ fileData: { fileUri: block.source.url } });
            }
          }
        }
      }
      if (parts.length > 0) contents.push({ role, parts });
    }
    if (Array.isArray(body.tools)) {
      for (const raw of body.tools) {
        if (isRecord(raw) && typeof raw.name === 'string') {
          tools.push({ name: raw.name, description: typeof raw.description === 'string' ? raw.description : '', parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } });
        }
      }
    }
    if (typeof body.max_tokens === 'number') maxTokens = body.max_tokens;
    if (typeof body.temperature === 'number') temperature = body.temperature;
    if (typeof body.top_p === 'number') topP = body.top_p;
    if (Array.isArray(body.stop_sequences)) stops = body.stop_sequences.filter((s) => typeof s === 'string');
  }

  const out: Json = { model, contents };
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  if (tools.length > 0) out.tools = [{ functionDeclarations: tools }];
  const config: Json = {};
  if (maxTokens !== undefined) config.maxOutputTokens = maxTokens;
  if (temperature !== undefined) config.temperature = temperature;
  if (topP !== undefined) config.topP = topP;
  if (stops.length > 0) config.stopSequences = stops;
  if (jsonMode) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}
