from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base

class Startup(Base):
    __tablename__ = "startups"

    id = Column(Integer, primary_key=True)
    name = Column(String)
    description = Column(String, default="" )
    website_url = Column(String, default="")
    created_at = Column(DateTime, default=datetime.utcnow)
    founders=Column(String, default="")
    stage=Column(String, default="")
    funding_amount=Column(Integer, default=0)
    funding_type=Column(String, default="")
    location=Column(String, default="")
    tags=Column(String, default="")
    # relationship
    articles = relationship("Article", back_populates="startup")


class Article(Base):
    __tablename__ = "articles"

    id = Column(Integer, primary_key=True)
    title = Column(String)
    link = Column(String)
    source = Column(String)
    published = Column(String)
    text = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Foreign key
    startup_id = Column(Integer, ForeignKey("startups.id"))

    # relationship
    startup = relationship("Startup", back_populates="articles")

class GithubRepo(Base):
    __tablename__ = "github_repos"
    id = Column(Integer, primary_key=True)
    title = Column(String)
    url= Column(String)
    stars=Column(Integer, default=0)
    language=Column(String, default="")
    description = Column(String, default="")
    created_at = Column(DateTime, default=datetime.utcnow)

    # Foreign key
    startup_id = Column(Integer, ForeignKey("startups.id"))

    # relationship
    startup = relationship("Startup", back_populates="articles")