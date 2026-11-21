import { NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';

const SYSTEM_TEXT =
  'You are Jamie, a sharp, concise AI financial advisor for a budgeting app. Give actionable budgeting advice in 1 to 2 punchy sentences.';

const SSE_HEADERS = {
  'Content-Type': 'text/event-stream',
  'Cache-Control': 'no-cache, no-transform',
  'Connection': 'keep-alive',
  'X-Accel-Buffering': 'no',
};

// Per-attempt fetch timeout (ms). Fail fast on 503 high-demand stalls.
const FETCH_TIMEOUT_MS = 5_000;

type Attempt = { model: string; retryOn?: number[] };

// gemini-3.8-flash first; on 429/503/404/timeout fall through immediately — no same-model retry.
// Fallback models do NOT start with 'gemini-3.8' so buildBody omits thinkingConfig (no 400 errors).
const FALLBACK_STATUSES = [429, 503, 404];
const ATTEMPTS: Attempt[] = [
  { model: 'gemini-3.8-flash' },
  { model: 'gemini-2.5-flash',      retryOn: FALLBACK_STATUSES },
  { model: 'gemini-3.5-flash-lite', retryOn: FALLBACK_STATUSES },
  { model: 'gemini-flash-latest',   retryOn: FALLBACK_STATUSES },
];

function extractUserPrompt(body: Record<string, unknown>): string | null {
  // { message: string }
  if (typeof body.message === 'string' && body.message.trim()) {
    return body.message.trim();
  }
  // { prompt: string }
  if (typeof body.prompt === 'string' && body.prompt.trim()) {
    return body.prompt.trim();
  }
  // { messages: Array<{ role, content }> } — take the last user entry
  if (Array.isArray(body.messages)) {
    for (let i = body.messages.length - 1; i >= 0; i--) {
      const m = body.messages[i];
      if (
        m &&
        typeof m === 'object' &&
        (m as Record<string, unknown>).role === 'user' &&
        typeof (m as Record<string, unknown>).content === 'string' &&
        ((m as Record<string, unknown>).content as string).trim()
      ) {
        return ((m as Record<string, unknown>).content as string).trim();
      }
    }
  }
  return null;
}

function buildBody(model: string, userPrompt: string): string {
  // thinkingConfig is only supported on the gemini-3.8 family
  const generationConfig: Record<string, unknown> = { maxOutputTokens: 100 };
  if (model.startsWith('gemini-3.8')) {
    generationConfig.thinkingConfig = { thinkingLevel: 'low' };
  }

  return JSON.stringify({
    contents: [{ role: 'user', parts: [{ text: userPrompt }] }],
    system_instruction: { parts: [{ text: SYSTEM_TEXT }] },
    generationConfig,
  });
}

export async function POST(req: Request) {
  try {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      return NextResponse.json(
        { reply: 'Error: GEMINI_API_KEY is missing in .env.local' },
        { status: 500 }
      );
    }

    const body = await req.json();
    const userPrompt = extractUserPrompt(body);

    if (!userPrompt) {
      return NextResponse.json(
        { reply: 'Error: no user message provided in the request body.' },
        { status: 400 }
      );
    }

    let lastStatus = 500;
    let lastErrorText = 'All model attempts failed.';

    for (let i = 0; i < ATTEMPTS.length; i++) {
      const attempt = ATTEMPTS[i];

      // Skip this slot if the previous failure isn't in its retryOn list
      if (i > 0 && attempt.retryOn && !attempt.retryOn.includes(lastStatus)) {
        continue;
      }

      // Per-attempt abort timeout so a hung upstream fails fast
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);

      let response: Response;
      try {
        response = await fetch(
          `https://generativelanguage.googleapis.com/v1beta/models/${attempt.model}:streamGenerateContent?alt=sse&key=${apiKey}`,
          {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: buildBody(attempt.model, userPrompt),
            signal: controller.signal,
          }
        );
      } catch (fetchErr: any) {
        clearTimeout(timer);
        // AbortError means the timeout fired — treat as 503 so we fall through
        lastStatus = fetchErr.name === 'AbortError' ? 503 : 500;
        lastErrorText = fetchErr.message ?? 'fetch failed';
        continue;
      }

      clearTimeout(timer);

      if (response.ok) {
        return new Response(response.body, { headers: SSE_HEADERS });
      }

      lastStatus = response.status;
      lastErrorText = await response.text();
    }

    return NextResponse.json(
      { reply: `API Error: ${lastErrorText}` },
      { status: lastStatus }
    );
  } catch (error: any) {
    return NextResponse.json({ reply: `Server Error: ${error.message}` }, { status: 500 });
  }
}
