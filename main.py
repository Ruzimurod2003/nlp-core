import json
import logging
import os
import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import APIKeyHeader
from core.anonymization_service import main_service
from core.ProcessDocumentViewModel import ProcessDocumentViewModel, ProcessDocumentResponseViewModel

description = (
    "SIA \"Corporate Solutions\" veidotais rīks kurš bāzēts uz mākslīgā intelekta (AI) bāzes."
)
logger = logging.getLogger(__file__)

app = FastAPI(
    title="Dabiskās valodas apstrāde",
    description=description,
    version="V1.0",
    docs_url="/"
)


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "appsettings.json")
    with open(config_path, "r") as config_file:
        con = json.load(config_file)
    return con


config = load_config()

API_KEY = os.getenv('API_KEY', config.get("API_KEY"))

api_key_header = APIKeyHeader(name="api-key", auto_error=False)


def get_api_key(key_header: str = Depends(api_key_header)):
    if key_header is None or key_header != API_KEY:
        raise HTTPException(status_code=401)
    return key_header


@app.post(
    path="/Api/Anonymization/ProcessDocument",
    tags=["Anonymization"],
    response_description="Atgriež datus",
    summary="Apstrādā dokumentu un atgriež atrastās entītes tajā.",
    response_model=ProcessDocumentResponseViewModel
)
async def process_document(
        view_model: ProcessDocumentViewModel,
        api_key: str = Depends(get_api_key)
):
    try:
        return main_service(view_model.text)
    except Exception as e:
        return {"error": str(e)}


def main():
    host = "localhost"
    port = 44334
    environment = "development"

    if environment == "development":
        uvicorn.run("main:app", host=host, port=port, reload=True)
    else:
        uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
