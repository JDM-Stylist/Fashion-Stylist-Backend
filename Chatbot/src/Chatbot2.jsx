import openai from "./config/openai.js";
import React,{useState,useEffect} from "react";
function Chatbot2(){
    const [chatHistory,setChatHistory] = useState([]);
    const [input,setInput] = useState("");
    async function Logic(userInput){
        try {
            const messages = chatHistory.map(([role,content])=>({role,content}));
            messages.push( { role: 'user', content: userInput } );
            const response = await openai.chat.completions.create({
                model:'gpt-3.5-turbo-16k-0613',
                messages:messages,
            });
            const completion = response.choices[0].message.content;
            return completion;
        } catch (error) {
            console.log(error);
            return "Sorry I caught an error. Please try again later.";
        }
    };
    async function handleInput(){
        if(input.trim() === ""){
            return;
        }
        let chat = '';
        chat = await Logic(input);
        updateHistory(input,chat);
        setInput('');
    }
    const updateHistory = (input,botresponse) =>{
        setChatHistory(previousHistory => {
            const newHistory = [...previousHistory];
            newHistory.push(['user',input]);
            newHistory.push(['assistant',botresponse]);
            return newHistory;
        })
    };
    useEffect(() => {
        function handleKeyDown(e) {
            if (e.key === 'Enter') {
                if (e.shiftKey) {
                    e.preventDefault(); // Prevent form submission
                    setInput(prevInput => prevInput + '\n'); // Add newline character to input
                } else {
                    e.preventDefault(); // Prevent form submission
                    handleInput();
                    setInput('');
                }
            }
        }
    
        window.addEventListener('keydown', handleKeyDown);
    
        return () => {
            window.removeEventListener('keydown', handleKeyDown);
        };
    }, [input, chatHistory]);
    
    function handleSubmit(e) {
        e.preventDefault();
        handleInput();
    }
    return (<div className="chat-container">
                <h1 style={{color: 'white'}}>Welcome to the chatbot programme</h1>
                <h2 style={{color:'white'}}>You can start chatting with the bot</h2>
                <div className="chat-history">
                    {chatHistory.map(([role, content], index) => (
                        <p key={index} className={role === 'user' ? 'user-message' : 'bot-message'}>
                            {role === 'user' ? 'You: ' : 'Bot: '}{content}
                        </p>
                    ))}
                </div>
                <form onSubmit={handleSubmit}>
                    <textarea
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        placeholder="Type here..."
                        style={{
                            marginRight: '5px',
                            width: '70%',
                            padding: '5px',
                            height: '17.5px', // Adjust the height as needed
                            verticalAlign: 'middle', // Center text vertically
                        }}
                    />
                    <button type="submit" className="send-btn">
                        Send
                    </button>
                </form>
            </div>
    );
}
export default Chatbot2;