from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    NOMIC_INFERENCE_MODE: str = 'local'  # set to 'remote' to use the API or 'local'
    GROQ_API_KEY: str
    GROQ_MODEL: str = 'llama3-70b-8192'  # https://console.groq.com/docs/models

    model_config = SettingsConfigDict(env_file='.env')

settings = Settings()