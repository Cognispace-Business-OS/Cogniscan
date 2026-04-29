from sqlalchemy import create_engine, Column, String, Integer, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime
from database import engine, Base

class Article(Base):
    __tablename__ = "articles"
    id = Column(Integer, primary_key=True)
    title = Column(String)
    link = Column(String)
    source = Column(String)
    published = Column(String)
    summary = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)