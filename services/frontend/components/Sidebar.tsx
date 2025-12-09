'use client';

import React from 'react';
import { MessageSquarePlus, History } from 'lucide-react';
import styles from './Sidebar.module.css';
import type { Message } from './ChatMessage';

export interface ChatHistory {
    id: string;
    title: string;
    timestamp: string;
    messages: Message[];
}

interface SidebarProps {
    chatHistory: ChatHistory[];
    currentChatId: string | null;
    onNewChat: () => void;
    onSelectChat: (chat: ChatHistory) => void;
}

export default function Sidebar({
    chatHistory,
    currentChatId,
    onNewChat,
    onSelectChat,
}: SidebarProps) {
    return (
        <aside className={styles.sidebar}>
            {/* Logo and Branding */}
            <div className={styles.branding}>
                <div className={styles.logo}>
                    <img src="/hc_logo.png" alt="Home Credit" className={styles.logoImg} />
                </div>
                <div className={styles.brandText}>
                    <h1 className={styles.brandTitle}>Home Credit</h1>
                    <p className={styles.brandTagline}>Empowering Smarter Credit Decisions</p>
                </div>
            </div>

            {/* New Chat Button */}
            <button className={styles.newChatBtn} onClick={onNewChat}>
                <MessageSquarePlus size={18} />
                <span>New Chat</span>
            </button>

            {/* Divider */}
            <div className={styles.divider} />

            {/* Chat History */}
            <div className={styles.historySection}>
                <div className={styles.historyHeader}>
                    <History size={14} />
                    <span>Recent Chats</span>
                </div>

                <div className={styles.historyList}>
                    {chatHistory.length === 0 ? (
                        <p className={styles.emptyHistory}>No chat history yet</p>
                    ) : (
                        chatHistory.slice(0, 20).map((chat) => (
                            <button
                                key={chat.id}
                                className={`${styles.historyItem} ${currentChatId === chat.id ? styles.historyItemActive : ''
                                    }`}
                                onClick={() => onSelectChat(chat)}
                            >
                                <span className={styles.historyTitle}>{chat.title}</span>
                                <span className={styles.historyTime}>{chat.timestamp}</span>
                            </button>
                        ))
                    )}
                </div>
            </div>

            {/* Footer */}
            <div className={styles.footer}>
                <p className={styles.footerText}>Powered by Gemini AI</p>
            </div>
        </aside>
    );
}
