"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const react_1 = require("react");
const App = () => {
    const [messages, setMessages] = (0, react_1.useState)(["Hi There"]);
    const wsref = (0, react_1.useRef)();
    (0, react_1.useEffect)(() => {
        const ws = new WebSocket("http://localhost:8080");
        ws.onmessage = (event) => {
            setMessages((msg) => [...msg, event.data]);
        };
        wsref.current = ws;
        ws.onopen();
    }, []);
    return (<div className="h-screen bg-black">
      <div className="h-[95vh]">
        {messages.map((msg) => (<div className="rounded bg-white p-4">
            <span>{msg}</span>
          </div>))}
      </div>
      <div className="bg-white flex p-4">
        <input id="message" type="text"/>
        <button onClick={() => {
            var _a;
            //@ts-ignore
            const message = (_a = document.getElementById("message")) === null || _a === void 0 ? void 0 : _a.value;
            wsref.current.send(JSON.stringify({
                type: "chat",
                payload: {
                    message: message,
                },
            }));
        }}>
          send
        </button>
      </div>
    </div>);
};
exports.default = App;
