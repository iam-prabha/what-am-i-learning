import { useEffect, useRef, useState } from "react";

const App = () => {
  const [messages, setMessages] = useState(["Hi There"]);
  const wsref = useRef();
  useEffect(() => {
    const ws = new WebSocket("http://localhost:8080");
    ws.onmessage = (event) => {
      setMessages((msg) => [...msg, event.data]);
    };
    wsref.current = ws;

    ws.onopen();
  }, []);
  return (
    <div className="h-screen bg-black">
      <div className="h-[95vh]">
        {messages.map((msg) => (
          <div className="rounded bg-white p-4">
            <span>{msg}</span>
          </div>
        ))}
      </div>
      <div className="bg-white flex p-4">
        <input id="message" type="text" />
        <button
          onClick={() => {
            //@ts-ignore
            const message = document.getElementById("message")?.value;
            wsref.current.send(
              JSON.stringify({
                type: "chat",
                payload: {
                  message: message,
                },
              })
            );
          }}
        >
          send
        </button>
      </div>
    </div>
  );
};
export default App;
