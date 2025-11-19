"""
FastAPI backend for FinAgent Multi-Agent System
Provides REST API endpoints for agent interactions
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, Dict, List
import pandas as pd
import uvicorn
import io
import logging
from pathlib import Path
import sys
import asyncio
import json
from datetime import datetime
import base64
# from contextlib import asynccontextmanager 
sys.path.append(str(Path(__file__).parent.parent)) 

from orchestrator import FinAgentOrchestrator
from agents.data_agent import DataAgent
from agents.insight_agent import InsightAgent
from agents.risk_agent import RiskAgent
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="FinAgent API",
    description="Multi-Agent Financial Insight Engine API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state for agents (in production, use proper state management)
orchestrator = None
current_data = None
streaming_task = None
streaming_queue = asyncio.Queue()

# Pydantic models for requests/responses
class AnalysisRequest(BaseModel):
    file_path: Optional[str] = None
    sample_size: Optional[int] = None


class InsightQuery(BaseModel):
    query: str
    

class RiskPrediction(BaseModel):
    transaction_data: Dict


class AnalysisResponse(BaseModel):
    status: str
    message: str
    summary: Optional[str] = None
    insights: Optional[Dict] = None
    risk_assessment: Optional[Dict] = None
    statistics: Optional[Dict] = None


class TransactionStream(BaseModel):
    transaction: Dict
    risk_score: float
    timestamp: str


class VisualizationRequest(BaseModel):
    visualization_type: str
    parameters: Optional[Dict] = None


@app.on_event("startup")
async def startup_event():
    """Initialize agents on startup"""
    global orchestrator
    
    logger.info(" Starting FinAgent API...")
    orchestrator = FinAgentOrchestrator(config=Config)
    orchestrator.compile()
    logger.info("✓ FinAgent initialized successfully")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "FinAgent Multi-Agent Financial Insight Engine API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "upload": "/api/upload",
            "analyze": "/api/analyze",
            "insights": "/api/insights",
            "risk": "/api/risk",
            "statistics": "/api/statistics",
            "streaming": "/api/stream/start",
            "visualizations": "/api/visualizations"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "orchestrator_ready": orchestrator is not None,
        "config": {
            "llm_provider": Config.LLM_PROVIDER,
            "embedding_model": Config.EMBEDDING_MODEL
        }
    }


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload transaction data file"""
    global current_data
    
    try:
        if file.filename is None:
            raise HTTPException(status_code=400, detail="File name is missing.")        
        contents = await file.read()
    
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        elif file.filename.endswith('.json'):
            df = pd.read_json(io.StringIO(contents.decode('utf-8')))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Use CSV or JSON.")
        
        current_data = df
        
        stats = {
            "filename": file.filename,
            "rows": len(df),
            "columns": list(df.columns),
            "size_mb": len(contents) / (1024 * 1024)
        }
        
        logger.info(f"✓ File uploaded: {file.filename} ({len(df)} rows)")
        
        return JSONResponse({
            "status": "success",
            "message": f"File {file.filename} uploaded successfully",
            "statistics": stats
        })
        
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_data(background_tasks: BackgroundTasks):
    """Run complete multi-agent analysis on uploaded data"""
    global current_data, orchestrator
    
    if current_data is None:
        raise HTTPException(status_code=400, detail="No data uploaded. Upload a file first.")
    
    if orchestrator is None:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")
    
    try:
        logger.info("🔄 Starting analysis...")
        
        result = orchestrator.run(data=current_data)
        
        response = AnalysisResponse(
            status="success",
            message="Analysis completed successfully",
            summary=result.get('summary'),
            insights=result.get('insights'),
            risk_assessment=result.get('risk_assessment'),
            statistics={
                'total_transactions': len(result.get('processed_data', [])),
                'anomalies_detected': len(result.get('anomalies', [])),
                'messages': result.get('messages', [])
            }
        )
        
        logger.info("✓ Analysis completed")
        return response
        
    except Exception as e:
        logger.error(f"Error in analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/insights")
async def generate_insights(query: InsightQuery):
    global current_data
    
    if current_data is None:
        raise HTTPException(status_code=400, detail="No data available")
    
    try:
        insight_agent = InsightAgent(config=Config)
        
        # Sample data for efficiency
        sample_size = min(1000, len(current_data))
        sample_df = current_data.sample(n=sample_size, random_state=42)
        
        # Ingest and generate insights
        insight_agent.ingest_data(sample_df)
        result = insight_agent.generate_insights(query.query)
        
        return JSONResponse({
            "status": "success",
            "query": query.query,
            "answer": result['answer'],
            "sources": result.get('source_documents', [])
        })
        
    except Exception as e:
        logger.error(f"Error generating insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/risk/summary")
async def get_risk_summary():
    """Get risk assessment summary"""
    global orchestrator, current_data
    
    # Fix: Check if orchestrator and risk agent are properly initialized
    if orchestrator is None:
        raise HTTPException(
            status_code=400, 
            detail="Orchestrator not initialized. Start the service properly."
        )
    
    if orchestrator.risk_agent is None or orchestrator.risk_agent.fraud_model is None:
        raise HTTPException(
            status_code=400, 
            detail="No risk model available. Run analysis first."
        )
    
    # Fix: Check if current_data is available
    if current_data is None:
        raise HTTPException(
            status_code=400, 
            detail="No data available. Upload data first."
        )
    
    try:
        # Fix: Ensure current_data is not None before calling predict_fraud
        if current_data is not None:
            predictions = orchestrator.risk_agent.predict_fraud(current_data)
            summary = orchestrator.risk_agent.get_risk_summary(predictions)
        else:
            raise HTTPException(
                status_code=400, 
                detail="No data available for risk assessment."
            )
        
        return JSONResponse({
            "status": "success",
            "risk_summary": summary
        })
        
    except Exception as e:
        logger.error(f"Error getting risk summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/statistics")
async def get_statistics():
    """Get dataset statistics"""
    global current_data
    
    if current_data is None:
        raise HTTPException(status_code=400, detail="No data available")
    
    try:
        data_agent = DataAgent()
        data_agent.df = current_data
        data_agent.processed_df = current_data
        
        stats = data_agent.get_statistics()
        
        return JSONResponse({
            "status": "success",
            "statistics": stats
        })
        
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/anomalies")
async def get_anomalies():
    """Get detected anomalies"""
    global orchestrator
    
    if orchestrator is None or not hasattr(orchestrator, 'data_agent'):
        raise HTTPException(status_code=400, detail="No analysis run yet")
    
    try:
        anomalies = orchestrator.data_agent.anomalies
        
        return JSONResponse({
            "status": "success",
            "count": len(anomalies),
            "anomalies": anomalies
        })
        
    except Exception as e:
        logger.error(f"Error getting anomalies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/data/sample")
async def get_sample_data(n: int = 10):
    """Get sample of current data"""
    global current_data
    
    if current_data is None:
        raise HTTPException(status_code=400, detail="No data available")
    
    try:
        sample = current_data.head(n)
        
        return JSONResponse({
            "status": "success",
            "sample": sample.to_dict(orient='records')
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Real-time streaming endpoints
@app.post("/api/stream/start")
async def start_streaming():
    """Start real-time transaction monitoring"""
    global orchestrator, streaming_task
    
    if orchestrator is None:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")
    
    try:
        # Start streaming in background
        if streaming_task is None or streaming_task.done():
            streaming_task = asyncio.create_task(_stream_transactions())
            logger.info("✓ Real-time streaming started")
        
        return JSONResponse({
            "status": "success",
            "message": "Real-time transaction monitoring started"
        })
        
    except Exception as e:
        logger.error(f"Error starting streaming: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stream/stop")
async def stop_streaming():
    """Stop real-time transaction monitoring"""
    global streaming_task
    
    try:
        if streaming_task and not streaming_task.done():
            streaming_task.cancel()
            logger.info("✓ Real-time streaming stopped")
        
        return JSONResponse({
            "status": "success",
            "message": "Real-time transaction monitoring stopped"
        })
        
    except Exception as e:
        logger.error(f"Error stopping streaming: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stream/status")
async def get_streaming_status():
    """Get real-time streaming status"""
    global streaming_task
    
    status = "inactive"
    if streaming_task:
        if not streaming_task.done():
            status = "active"
        else:
            status = "stopped"
    
    return JSONResponse({
        "status": "success",
        "streaming_status": status,
        "queue_size": streaming_queue.qsize()
    })


async def _stream_transactions():
    """Background task to simulate transaction streaming"""
    global orchestrator, streaming_queue
    
    try:
        # Simulate transaction stream
        merchants = ['Amazon', 'Walmart', 'Target', 'Starbucks', 'Shell', 'Casino']
        categories = ['Online', 'Retail', 'Food', 'Gas', 'Gambling']
        
        transaction_id = 0
        while True:
            # Generate a mock transaction
            transaction = {
                'transaction_id': transaction_id,
                'amount': float(pd.np.random.exponential(50)),
                'merchant': pd.np.random.choice(merchants),
                'category': pd.np.random.choice(categories),
                'hour': int(pd.np.random.randint(0, 24)),
                'day_of_week': int(pd.np.random.randint(0, 7))
            }
            
            # Occasionally create high-risk transactions
            if pd.np.random.random() < 0.05:
                transaction['amount'] = float(pd.np.random.exponential(5000))
                transaction['merchant'] = 'Casino'
                transaction['category'] = 'Gambling'
                transaction['hour'] = int(pd.np.random.choice([2, 3, 4, 5]))
            
            # Assess risk using the risk agent if available
            risk_score = 0.1  # Default low risk
            if orchestrator and orchestrator.risk_agent and orchestrator.risk_agent.fraud_model:
                try:
                    df = pd.DataFrame([transaction])
                    risk_score = float(orchestrator.risk_agent.fraud_model.predict_proba(df)[0][1])
                except Exception as e:
                    logger.warning(f"Error assessing risk for streaming transaction: {e}")
                    # Fallback to heuristic scoring
                    risk_score = _heuristic_risk_score(transaction)
            else:
                # Fallback to heuristic scoring
                risk_score = _heuristic_risk_score(transaction)
            
            # Create stream item
            stream_item = TransactionStream(
                transaction=transaction,
                risk_score=risk_score,
                timestamp=datetime.now().isoformat()
            )
            
            # Add to queue
            await streaming_queue.put(stream_item)
            
            transaction_id += 1
            await asyncio.sleep(0.1)  # Simulate real-time arrival
            
    except asyncio.CancelledError:
        logger.info("Streaming task cancelled")
    except Exception as e:
        logger.error(f"Error in streaming task: {e}")


def _heuristic_risk_score(transaction: Dict) -> float:
    """
    Simple heuristic-based risk scoring for streaming transactions
    """
    risk_score = 0.0
    
    # Amount-based risk
    amount = transaction.get('amount', 0)
    if amount > 10000:
        risk_score += 0.4
    elif amount > 5000:
        risk_score += 0.2
    elif amount > 1000:
        risk_score += 0.1
        
    # Time-based risk (unusual hours)
    hour = transaction.get('hour', 12)
    if hour < 6 or hour > 22:
        risk_score += 0.2
        
    # Merchant-based risk (if we have a list of high-risk merchants)
    high_risk_merchants = ['casino', 'gambling', 'adult']
    merchant = str(transaction.get('merchant', '')).lower()
    if any(risky in merchant for risky in high_risk_merchants):
        risk_score += 0.3
        
    # Category-based risk
    high_risk_categories = ['cash', 'gambling', 'adult']
    category = str(transaction.get('category', '')).lower()
    if any(risky in category for risky in high_risk_categories):
        risk_score += 0.2
        
    # Ensure score is between 0 and 1
    return min(1.0, max(0.0, risk_score))


@app.get("/api/stream/events")
async def stream_events():
    """Stream real-time transaction events"""
    async def event_generator():
        while True:
            try:
                # Wait for next item in queue
                item = await asyncio.wait_for(streaming_queue.get(), timeout=1.0)
                yield f"data: {json.dumps(item.dict())}\n\n"
            except asyncio.TimeoutError:
                # Send keep-alive
                yield f"data: {json.dumps({'type': 'keepalive', 'timestamp': datetime.now().isoformat()})}\n\n"
            except Exception as e:
                logger.error(f"Error in event stream: {e}")
                break
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")


# Visualization endpoints
@app.post("/api/visualizations/generate")
async def generate_visualizations(request: VisualizationRequest):
    """Generate specific visualizations"""
    global orchestrator, current_data
    
    if current_data is None:
        raise HTTPException(status_code=400, detail="No data available")
    
    if orchestrator is None:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")
    
    try:
        # Run analysis with visualization request
        result = orchestrator.run(data=current_data, user_query="Generate advanced visualizations")
        
        # Get visualizations from result
        visualizations = result.get('advanced_visualizations', {})
        
        if not visualizations:
            raise HTTPException(status_code=404, detail="No visualizations generated")
        
        return JSONResponse({
            "status": "success",
            "visualizations": list(visualizations.keys()),
            "message": f"Generated {len(visualizations)} visualizations"
        })
        
    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/visualizations/{viz_name}")
async def get_visualization(viz_name: str):
    """Get a specific visualization by name"""
    global orchestrator, current_data
    
    if current_data is None:
        raise HTTPException(status_code=400, detail="No data available")
    
    if orchestrator is None:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")
    
    try:
        # Run analysis with visualization request if not already done
        result = orchestrator.run(data=current_data, user_query="Generate advanced visualizations")
        
        # Get visualizations from result
        visualizations = result.get('advanced_visualizations', {})
        
        if viz_name not in visualizations:
            raise HTTPException(status_code=404, detail=f"Visualization '{viz_name}' not found")
        
        viz_data = visualizations[viz_name]
        
        # Check if it's an interactive visualization (HTML) or static image
        if viz_data.startswith('<'):
            # Interactive visualization (HTML)
            return HTMLResponse(content=viz_data, status_code=200)
        else:
            # Static image (base64)
            return JSONResponse({
                "status": "success",
                "visualization_name": viz_name,
                "data": viz_data,
                "type": "image"
            })
        
    except Exception as e:
        logger.error(f"Error getting visualization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/visualizations/list")
async def list_visualizations():
    """List all available visualizations"""
    global orchestrator, current_data
    
    if current_data is None:
        return JSONResponse({
            "status": "success",
            "visualizations": [],
            "message": "No data available - upload data to generate visualizations"
        })
    
    if orchestrator is None:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")
    
    try:
        # Run analysis with visualization request if not already done
        result = orchestrator.run(data=current_data, user_query="Generate advanced visualizations")
        
        # Get visualizations from result
        visualizations = result.get('advanced_visualizations', {})
        
        return JSONResponse({
            "status": "success",
            "visualizations": list(visualizations.keys()),
            "count": len(visualizations)
        })
        
    except Exception as e:
        logger.error(f"Error listing visualizations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Run the API
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=Config.API_HOST,
        port=Config.API_PORT,
        reload=True,
        log_level="info"
    )