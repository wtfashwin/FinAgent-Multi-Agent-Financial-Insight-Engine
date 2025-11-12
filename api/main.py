"""
FastAPI backend for FinAgent Multi-Agent System
Provides REST API endpoints for agent interactions
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, List
import pandas as pd
import uvicorn
import io
import logging
from pathlib import Path
import sys

# Add parent directory to path
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


@app.on_event("startup")
async def startup_event():
    """Initialize agents on startup"""
    global orchestrator
    
    logger.info("🚀 Starting FinAgent API...")
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
            "statistics": "/api/statistics"
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
        # Read uploaded file
        contents = await file.read()
        
        # Detect file type and read accordingly
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        elif file.filename.endswith('.json'):
            df = pd.read_json(io.StringIO(contents.decode('utf-8')))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Use CSV or JSON.")
        
        # Store data globally (in production, use database)
        current_data = df
        
        # Get basic statistics
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
        
        # Run orchestrator
        result = orchestrator.run(data=current_data)
        
        # Extract results
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
    """Generate insights based on a specific query"""
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
    global orchestrator
    
    if orchestrator is None or orchestrator.risk_agent.fraud_model is None:
        raise HTTPException(
            status_code=400, 
            detail="No risk model available. Run analysis first."
        )
    
    try:
        predictions = orchestrator.risk_agent.predict_fraud(current_data)
        summary = orchestrator.risk_agent.get_risk_summary(predictions)
        
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


# Run the API
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=Config.API_HOST,
        port=Config.API_PORT,
        reload=True,
        log_level="info"
    )