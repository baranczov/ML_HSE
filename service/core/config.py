from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Model
    model_path: str = ""               # empty → uses default from resolve_weights_path
    backbone_name: str = "resnet18"
    image_size: int = 224
    cache_size: int = 10_000

    # Upload limits
    max_upload_bytes: int = 5 * 1024 * 1024  # 5 MB
    max_batch_size: int = 8

    # Concurrency: max simultaneous inference forward passes
    max_concurrency: int = 4

    debug: bool = False


settings = Settings()
