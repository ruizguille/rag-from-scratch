from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    NOMIC_INFERENCE_MODE: str = 'remote'  # set to 'remote' to use the API or 'local'
    GROQ_API_KEY: str

    model_config = SettingsConfigDict(env_file='.env')

settings = Settings()