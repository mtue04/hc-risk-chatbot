'use client';

import React, { useState, useRef, useEffect } from 'react';
import { Plus, Send, Paperclip, Mic, X, MicOff, Loader2 } from 'lucide-react';
import styles from './ChatInput.module.css';

interface ChatInputProps {
    onSendMessage: (message: string, file?: File, audio?: Blob) => void;
    isLoading?: boolean;
    disabled?: boolean;
}

export default function ChatInput({ onSendMessage, isLoading, disabled }: ChatInputProps) {
    const [message, setMessage] = useState('');
    const [showPopover, setShowPopover] = useState(false);
    const [isRecording, setIsRecording] = useState(false);
    const [recordingTime, setRecordingTime] = useState(0);
    const [pendingFile, setPendingFile] = useState<File | null>(null);

    const textareaRef = useRef<HTMLTextAreaElement>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const popoverRef = useRef<HTMLDivElement>(null);
    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const audioChunksRef = useRef<Blob[]>([]);
    const timerRef = useRef<NodeJS.Timeout | null>(null);

    // Auto-resize textarea
    useEffect(() => {
        if (textareaRef.current) {
            textareaRef.current.style.height = 'auto';
            textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 150)}px`;
        }
    }, [message]);

    // Close popover when clicking outside
    useEffect(() => {
        function handleClickOutside(event: MouseEvent) {
            if (popoverRef.current && !popoverRef.current.contains(event.target as Node)) {
                setShowPopover(false);
            }
        }
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    // Handle send
    const handleSend = () => {
        if ((message.trim() || pendingFile) && !isLoading) {
            onSendMessage(message.trim(), pendingFile || undefined);
            setMessage('');
            setPendingFile(null);
        }
    };

    // Handle key press
    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    // Handle file upload
    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            setPendingFile(file);
            setShowPopover(false);
        }
    };

    // Start recording
    const startRecording = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
            mediaRecorderRef.current = mediaRecorder;
            audioChunksRef.current = [];

            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunksRef.current.push(event.data);
                }
            };

            mediaRecorder.onstop = () => {
                const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
                onSendMessage('', undefined, audioBlob);
                stream.getTracks().forEach(track => track.stop());
            };

            mediaRecorder.start();
            setIsRecording(true);
            setShowPopover(false);

            // Start timer
            timerRef.current = setInterval(() => {
                setRecordingTime(prev => prev + 1);
            }, 1000);
        } catch (error) {
            console.error('Failed to start recording:', error);
            alert('Could not access microphone. Please check permissions.');
        }
    };

    // Stop recording
    const stopRecording = () => {
        if (mediaRecorderRef.current && isRecording) {
            mediaRecorderRef.current.stop();
            setIsRecording(false);
            setRecordingTime(0);
            if (timerRef.current) {
                clearInterval(timerRef.current);
            }
        }
    };

    // Cancel recording
    const cancelRecording = () => {
        if (mediaRecorderRef.current && isRecording) {
            mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
            setIsRecording(false);
            setRecordingTime(0);
            if (timerRef.current) {
                clearInterval(timerRef.current);
            }
        }
    };

    // Format recording time
    const formatTime = (seconds: number) => {
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    };

    // Remove pending file
    const removePendingFile = () => {
        setPendingFile(null);
        if (fileInputRef.current) {
            fileInputRef.current.value = '';
        }
    };

    return (
        <div className={styles.container}>
            {/* Pending File Preview */}
            {pendingFile && (
                <div className={styles.filePreview}>
                    <div className={styles.fileInfo}>
                        <Paperclip size={14} />
                        <span>{pendingFile.name}</span>
                    </div>
                    <button className={styles.removeFile} onClick={removePendingFile}>
                        <X size={14} />
                    </button>
                </div>
            )}

            {/* Recording UI */}
            {isRecording ? (
                <div className={styles.recordingContainer}>
                    <div className={styles.recordingIndicator}>
                        <span className={styles.recordingDot} />
                        <span className={styles.recordingTime}>{formatTime(recordingTime)}</span>
                        <span className={styles.recordingText}>Recording...</span>
                    </div>
                    <div className={styles.recordingActions}>
                        <button className={styles.cancelBtn} onClick={cancelRecording}>
                            <X size={18} />
                            <span>Cancel</span>
                        </button>
                        <button className={styles.stopBtn} onClick={stopRecording}>
                            <MicOff size={18} />
                            <span>Send</span>
                        </button>
                    </div>
                </div>
            ) : (
                /* Normal Input UI */
                <div className={styles.inputWrapper}>
                    {/* Plus/Attachment Button */}
                    <div className={styles.popoverContainer} ref={popoverRef}>
                        <button
                            className={`${styles.attachBtn} ${showPopover ? styles.attachBtnActive : ''}`}
                            onClick={() => setShowPopover(!showPopover)}
                            disabled={disabled || isLoading}
                        >
                            <Plus size={20} className={showPopover ? styles.rotated : ''} />
                        </button>

                        {/* Popover Menu */}
                        {showPopover && (
                            <div className={styles.popover}>
                                <button className={styles.popoverItem} onClick={() => fileInputRef.current?.click()}>
                                    <Paperclip size={18} />
                                    <span>Upload file</span>
                                </button>
                                <button className={styles.popoverItem} onClick={startRecording}>
                                    <Mic size={18} />
                                    <span>Voice input</span>
                                </button>
                            </div>
                        )}
                    </div>

                    {/* Text Input */}
                    <textarea
                        ref={textareaRef}
                        className={styles.textarea}
                        placeholder="Ask about credit risk..."
                        value={message}
                        onChange={(e) => setMessage(e.target.value)}
                        onKeyDown={handleKeyDown}
                        disabled={disabled || isLoading}
                        rows={1}
                    />

                    {/* Send Button */}
                    <button
                        className={`${styles.sendBtn} ${(message.trim() || pendingFile) && !isLoading ? styles.sendBtnActive : ''}`}
                        onClick={handleSend}
                        disabled={(!message.trim() && !pendingFile) || isLoading}
                    >
                        {isLoading ? (
                            <Loader2 size={20} className={styles.spinning} />
                        ) : (
                            <Send size={20} />
                        )}
                    </button>

                    {/* Hidden File Input */}
                    <input
                        ref={fileInputRef}
                        type="file"
                        className={styles.hiddenInput}
                        accept=".png,.jpg,.jpeg,.pdf,.txt,.docx,.doc"
                        onChange={handleFileChange}
                    />
                </div>
            )}
        </div>
    );
}
