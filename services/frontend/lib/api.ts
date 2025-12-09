// API calls go through Next.js API routes which proxy to the chatbot service
// This avoids CORS issues and allows server-side environment variables

export interface ChatResponse {
    answer: string;
    session_id: string;
    applicant_id?: number;
    risk_probability?: number;
    tool_outputs?: any[];
    conversation_history?: { role: string; content: string }[];
}

export interface StreamEvent {
    type: 'ai_message' | 'tool_result' | 'done' | 'error' | 'analysis_started' | 'analysis_step';
    content?: string;
    has_tool_calls?: boolean;
    tool_calls?: Array<{
        name: string;
        args: any;
        id: string;
    }>;
    tool_call_id?: string;
    result?: any;
    session_id?: string;
    thread_id?: string;
    error?: string;
    step_result?: {
        step_number: number;
        description?: string;
        data_summary?: string;
        chart_type: string;
        chart_image_base64?: string;
        insights: string;
    };
}

export interface ChatRequest {
    question: string;
    session_id?: string;
    applicant_id?: number;
}

/**
 * Send a text message to the chatbot
 */
export async function sendMessage(
    question: string,
    sessionId?: string,
    applicantId?: number
): Promise<ChatResponse> {
    const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            question,
            session_id: sessionId,
            applicant_id: applicantId,
        }),
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.error || `API error: ${response.status}`);
    }

    return response.json();
}

/**
 * Send a multimodal message (with audio or file)
 */
export async function sendMultimodal(
    question: string,
    sessionId?: string,
    audio?: Blob,
    file?: File
): Promise<ChatResponse> {
    const formData = new FormData();

    if (question) {
        formData.append('question', question);
    }
    if (sessionId) {
        formData.append('session_id', sessionId);
    }
    if (audio) {
        formData.append('audio', audio, 'voice.webm');
    }
    if (file) {
        formData.append('image', file, file.name);
    }

    const response = await fetch('/api/chat/multimodal', {
        method: 'POST',
        body: formData,
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.error || `API error: ${response.status}`);
    }

    return response.json();
}

/**
 * Send a message with streaming response
 * Calls the callback with each event as it arrives
 */
export async function sendMessageStream(
    question: string,
    sessionId: string | undefined,
    applicantId: number | undefined,
    onEvent: (event: StreamEvent) => void
): Promise<void> {
    const response = await fetch('http://localhost:8500/chat/stream', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            question,
            session_id: sessionId,
            applicant_id: applicantId,
        }),
    });

    if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
    }

    const reader = response.body?.getReader();
    if (!reader) {
        throw new Error('Response body is not readable');
    }

    const decoder = new TextDecoder();
    let buffer = '';

    try {
        while (true) {
            const { done, value } = await reader.read();

            if (done) break;

            // Decode the chunk and add to buffer
            buffer += decoder.decode(value, { stream: true });

            // Process complete SSE messages (separated by \n\n)
            const lines = buffer.split('\n\n');
            buffer = lines.pop() || ''; // Keep incomplete message in buffer

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = line.slice(6); // Remove 'data: ' prefix
                    try {
                        const event: StreamEvent = JSON.parse(data);
                        onEvent(event);
                    } catch (e) {
                        console.error('Failed to parse SSE event:', data, e);
                    }
                }
            }
        }
    } finally {
        reader.releaseLock();
    }
}

// Note: These functions are not currently used but kept for future use
// They would need API route proxies if enabled

// export async function clearConversation(sessionId: string): Promise<void> {
//     await fetch(`/api/conversation/${sessionId}`, {
//         method: 'DELETE',
//     });
// }

// export async function getConversationHistory(
//     sessionId: string
// ): Promise<{ role: string; content: string }[]> {
//     const response = await fetch(`/api/conversation/${sessionId}`);
//     if (!response.ok) {
//         throw new Error(`API error: ${response.status}`);
//     }
//     const data = await response.json();
//     return data.history || [];
// }
