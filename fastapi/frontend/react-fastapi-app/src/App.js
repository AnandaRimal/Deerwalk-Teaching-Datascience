import React, { useState } from "react";
import './App.css';
function App() {

  const [message, setMessage] = useState("");

  const getMessage = async () => {
    const response = await fetch("");
    const data = await response.json();
    setMessage(data.message);
  };

  return (
    <div style={{textAlign:"center", marginTop:"50px"}}>

      <h1>FastAPI + React Example</h1>

      <button onClick={getMessage}>
        Get Message from API
      </button>

      <h2>{message}</h2>

    </div>
  );
}

export default App;