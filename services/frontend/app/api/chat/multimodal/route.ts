import { NextRequest, NextResponse } from 'next/server';

const CHATBOT_API = process.env.CHATBOT_API_URL || 'http://chatbot:8500';

export async function POST(request: NextRequest) {
    try {
        const formData = await request.formData();

        // Forward the form data to the chatbot API
        const response = await fetch(`${CHATBOT_API}/chat/multimodal`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            return NextResponse.json(
                { error: `Chatbot API error: ${response.status}` },
                { status: response.status }
            );
        }

        const data = await response.json();
        return NextResponse.json(data);
    } catch (error) {
        console.error('Multimodal proxy error:', error);
        return NextResponse.json(
            { error: 'Failed to connect to chatbot service' },
            { status: 500 }
        );
    }
}
