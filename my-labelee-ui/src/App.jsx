// src/App.jsx
import React, { useState } from 'react';
import './App.css';

function App() {
  const [imageUrl, setImageUrl] = useState('');
  const [textPrompt, setTextPrompt] = useState('');
  
  // --- NEW: More detailed state ---
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handlePrediction = async () => {
    // Reset state for a new request
    setIsLoading(true);
    setError('');
    setResult(null);

    try {
      const response = await fetch('http://127.0.0.1:8000/api/v1/predict/my-test-model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_url: imageUrl,
          text_prompt: textPrompt,
        })
      });

      // --- NEW: Handle potential errors from the API ---
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'An unknown error occurred.');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <>
      <h1>Labelee AI</h1>
      <h2>Prediction Playground</h2>
      <div className="predictor-container">
        <div className="input-group">
          <label htmlFor="image-url">Image URL</label>
          <input
            id="image-url"
            type="text"
            value={imageUrl}
            onChange={(e) => setImageUrl(e.target.value)}
            placeholder="https://path.to/your/image.jpg"
          />
        </div>
        <div className="input-group">
          <label htmlFor="text-prompt">Text Prompt</label>
          <textarea
            id="text-prompt"
            value={textPrompt}
            onChange={(e) => setTextPrompt(e.target.value)}
            placeholder="A description of the image..."
          />
        </div>
        <button onClick={handlePrediction} disabled={isLoading}>
          {isLoading ? 'Thinking...' : 'Get Similarity Score'}
        </button>
      </div>
      
      {/* --- NEW: Conditional rendering for loading, error, or success --- */}
      {isLoading && <p className="loading">Communicating with the AI...</p>}
      
      {error && <div className="error">Error: {error}</div>}
      
      {result && (
        <div className="result">
          <h2>Similarity Score: {result.score.toFixed(4)}</h2>
        </div>
      )}
    </>
  );
}

export default App;