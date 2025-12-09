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
