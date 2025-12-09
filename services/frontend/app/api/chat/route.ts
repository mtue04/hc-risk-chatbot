import { NextRequest, NextResponse } from 'next/server';

const CHATBOT_API = process.env.CHATBOT_API_URL || 'http://chatbot:8500';

export async function POST(request: NextRequest) {
    try {
        const body = await request.json();

        const response = await fetch(`${CHATBOT_API}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(body),
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
        console.error('Chat proxy error:', error);
        return NextResponse.json(
            { error: 'Failed to connect to chatbot service' },
            { status: 500 }
        );
    }
}
