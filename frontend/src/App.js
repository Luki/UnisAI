import React, { useEffect, useMemo, useRef, useState } from "react";
import axios from "axios";
import "./App.css";
import {
  Card,
  CardHeader,
  CardContent,
  CardFooter,
  CardTitle,
} from "./components/ui/card";
import { Button } from "./components/ui/button";
import { Input } from "./components/ui/input";
import { ScrollArea } from "./components/ui/scroll-area";
import { Avatar, AvatarImage, AvatarFallback } from "./components/ui/avatar";
import { Separator } from "./components/ui/separator";
import { Tooltip, TooltipProvider, TooltipTrigger, TooltipContent } from "./components/ui/tooltip";
import { Send, Sparkles, Loader2 } from "lucide-react";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const MessageBubble = ({ isUser, text, time }) => {
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      {!isUser && (
        <Avatar className="mr-3 mt-1 h-8 w-8 shadow-sm">
          <AvatarImage src="https://api.dicebear.com/9.x/bottts-neutral/svg?seed=UnisAI" alt="Unis" />
          <AvatarFallback>U</AvatarFallback>
        </Avatar>
      )}
      <div
        className={`max-w-[70%] rounded-2xl px-4 py-3 text-sm leading-relaxed shadow-sm ${
          isUser
            ? "bg-[hsl(210,20%,97%)] text-[hsl(215,25%,15%)] border border-[hsl(214,16%,92%)]"
            : "bg-white text-[hsl(215,25%,15%)] border border-[hsl(214,16%,92%)]"
        }`}
      >
        <div className="whitespace-pre-wrap">{text}</div>
        <div className="mt-2 text-[11px] text-[hsl(215,16%,45%)]">{time}</div>
      </div>
      {isUser && (
        <Avatar className="ml-3 mt-1 h-8 w-8 shadow-sm">
          <AvatarImage src="https://api.dicebear.com/9.x/initials/svg?seed=You" alt="You" />
          <AvatarFallback>Y</AvatarFallback>
        </Avatar>
      )}
    </div>
  );
};

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const endRef = useRef(null);

  const helloWorldApi = async () => {
    try {
      await axios.get(`${API}/`);
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    helloWorldApi();
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const disabled = useMemo(() => loading || !input.trim(), [loading, input]);

  const onSend = async () => {
    if (!input.trim()) return;
    const userText = input;
    setInput("");
    setMessages((prev) => [
      ...prev,
      { id: `user_${Date.now()}`, isUser: true, text: userText, time: new Date().toLocaleTimeString() },
    ]);

    setLoading(true);
    try {
      const res = await axios.post(`${API}/chat`, {
        message: userText,
        session_id: sessionId,
      });
      const data = res.data;
      if (!sessionId) setSessionId(data.session_id);
      setMessages((prev) => [
        ...prev,
        {
          id: data.id,
          isUser: false,
          text: data.response,
          time: new Date(data.timestamp).toLocaleTimeString(),
        },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          id: `err_${Date.now()}`,
          isUser: false,
          text: "Error: unable to get a response. Please try again.",
          time: new Date().toLocaleTimeString(),
        },
      ]);
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[hsl(210,20%,98%)]">
      <div className="mx-auto max-w-4xl px-6 pt-12 pb-24">
        <div className="mb-6">
          <h1 className="text-3xl font-semibold tracking-tight text-[hsl(220,20%,20%)]">Unis AI</h1>
          <p className="mt-2 text-[hsl(215,16%,45%)]">Ask anything. Powered by GPT-4.1</p>
        </div>

        <Card className="border border-[hsl(214,16%,92%)]/80 shadow-[0_10px_40px_-12px_rgba(16,24,40,0.08)] backdrop-blur-[8px]">
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle className="flex items-center gap-2 text-[hsl(220,20%,20%)]">
              <Sparkles className="h-5 w-5 text-[hsl(220,65%,55%)]" />
              Chat with Unis
            </CardTitle>
          </CardHeader>
          <Separator />
          <CardContent className="p-0">
            <ScrollArea className="h-[480px] px-6 py-6">
              <div className="flex flex-col gap-4">
                {messages.length === 0 && (
                  <div className="text-sm text-[hsl(215,16%,45%)]">
                    Tip: Ask about anything — for example, "Summarize this concept" or "Draft a friendly email".
                  </div>
                )}
                {messages.map((m) => (
                  <MessageBubble key={m.id} isUser={m.isUser} text={m.text} time={m.time} />
                ))}
                <div ref={endRef} />
              </div>
            </ScrollArea>
          </CardContent>
          <Separator />
          <CardFooter>
            <div className="flex w-full items-center gap-3">
              <Input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Type your question..."
                className="h-11 rounded-full border-[hsl(214,16%,92%)] bg-white focus-visible:ring-[hsl(220,65%,55%)]"
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    onSend();
                  }
                }}
              />
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <span>
                      <Button
                        onClick={onSend}
                        disabled={disabled}
                        className="h-11 rounded-full bg-[hsl(220,65%,55%)] text-white hover:bg-[hsl(220,70%,45%)] disabled:opacity-60"
                      >
                        {loading ? (
                          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        ) : (
                          <Send className="mr-2 h-4 w-4" />
                        )}
                        Send
                      </Button>
                    </span>
                  </TooltipTrigger>
                  <TooltipContent>Send message</TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </div>
          </CardFooter>
        </Card>
      </div>
    </div>
  );
}

export default App;