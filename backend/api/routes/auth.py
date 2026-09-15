from urllib.parse import urlencode

import requests
from config import settings
from database import User, get_db
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from itsdangerous import URLSafeTimedSerializer
from sqlalchemy.orm import Session

from api.schemas import UserOut

router = APIRouter(prefix="/auth", tags=["auth"])


def _make_session_token(user_id: int) -> str:
    s = URLSafeTimedSerializer(settings.secret_key, salt="session")
    return s.dumps({"user_id": user_id})


def _resolve_user(request: Request, db: Session) -> User | None:
    token = request.cookies.get("session")
    if not token:
        return None
    s = URLSafeTimedSerializer(settings.secret_key, salt="session")
    try:
        data = s.loads(token, max_age=86400 * 7)
        return db.query(User).filter(User.id == data["user_id"]).first()
    except Exception:
        return None


@router.get("/me")
def get_me(request: Request, db: Session = Depends(get_db)):
    user = _resolve_user(request, db)
    if not user:
        return None
    return UserOut.model_validate(user)


@router.get("/steam/login")
def steam_login(request: Request):
    if not settings.steam_api_key:
        raise HTTPException(status_code=503, detail="Steam login not configured")

    params = {
        "openid.ns": "http://specs.openid.net/auth/2.0",
        "openid.mode": "checkid_setup",
        "openid.return_to": f"{settings.api_url}/auth/callback",
        "openid.realm": settings.frontend_url,
        "openid.identity": "http://specs.openid.net/auth/2.0/identifier_select",
        "openid.claimed_id": "http://specs.openid.net/auth/2.0/identifier_select",
    }
    steam_login_url = "https://steamcommunity.com/openid/login?" + urlencode(params)
    return {"login_url": steam_login_url}


@router.get("/callback")
def steam_callback(
    request: Request,
    db: Session = Depends(get_db),
):
    openid_identity = request.query_params.get("openid.claimed_id")
    if not openid_identity:
        raise HTTPException(status_code=400, detail="Missing OpenID identity")

    steam_id = openid_identity.rstrip("/").split("/")[-1]
    if not steam_id or not steam_id.isdigit():
        raise HTTPException(status_code=400, detail="Invalid Steam ID")

    if not settings.steam_api_key:
        raise HTTPException(status_code=503, detail="Steam login not configured")

    player_url = (
        "http://api.steampowered.com/ISteamUser/GetPlayerSummaries/v0002/"
        f"?key={settings.steam_api_key}&steamids={steam_id}"
    )
    try:
        resp = requests.get(player_url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        player = data["response"]["players"][0]
        username = player.get("personaname")
        avatar_url = player.get("avatarfull")
    except Exception:
        username = None
        avatar_url = None

    user = db.query(User).filter(User.steam_id == steam_id).first()
    if not user:
        user = User(
            steam_id=steam_id,
            username=username,
            avatar_url=avatar_url,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    else:
        if username:
            user.username = username
            user.avatar_url = avatar_url
        db.commit()

    token = _make_session_token(user.id)

    # The session travels in the httponly cookie set below; keep it out of
    # the URL so it is not stored in browser history or server logs.
    redirect_url = f"{settings.frontend_url}/portfolio"
    from fastapi.responses import RedirectResponse

    resp = RedirectResponse(url=redirect_url)
    resp.set_cookie(
        key="session",
        value=token,
        max_age=86400 * 7,
        httponly=True,
        samesite="lax",
        secure=settings.is_production(),
    )
    return resp


@router.post("/logout")
def logout(response: Response):
    response.delete_cookie("session", path="/")
    return {"ok": True}
