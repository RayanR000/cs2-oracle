from config import settings
from database import User, get_db
from fastapi import APIRouter, Depends, HTTPException, Request
from itsdangerous import URLSafeTimedSerializer
from sqlalchemy.orm import Session

router = APIRouter(prefix="/portfolio", tags=["portfolio"])


def _get_current_user(request: Request, db: Session) -> User | None:
    token = request.cookies.get("session")
    if not token:
        return None
    s = URLSafeTimedSerializer(settings.secret_key, salt="session")
    try:
        data = s.loads(token, max_age=86400 * 7)
        return db.query(User).filter(User.id == data["user_id"]).first()
    except Exception:
        return None


@router.get("/inventory")
def get_inventory(request: Request, db: Session = Depends(get_db)):
    user = _get_current_user(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    # TODO: Integrate with Steam Inventory API
    # For now, return an empty inventory
    return {
        "steam_id": user.steam_id,
        "username": user.username,
        "items": [],
    }
