import { WebSocketServer } from "ws";

const wss = new WebSocketServer({ port: 8080 });

//event handler
wss.on("connection", (socket) => {
  console.log("A User  connected");
  setInterval(() => {
    socket.send("current process stock: " + Math.random());
  }, 500);

  socket.on("message", (e) => {
    if (e.toString() === "ping") {
      socket.send("pong");
    }
  });
});
