# simple_login.py
# Login sencillo por contraseña única para Streamlit
# - Lee la contraseña válida de st.secrets["APP_PASSWORD"] o de la variable de entorno APP_PASSWORD
# - Mantiene sesión en st.session_state
# - No requiere dependencias adicionales

from __future__ import annotations
import os
import hmac
import streamlit as st

def _expected_password(key: str) -> str:
    try:
        # st.secrets tiene prioridad
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return str(os.environ.get(key, ""))

def require_login(
    app_name: str = "App",
    password_key: str = "APP_PASSWORD",
    session_flag: str = "__is_auth",
) -> bool:
    """
    Muestra un formulario de acceso en la barra lateral y devuelve True/False.
    - app_name: etiqueta informativa
    - password_key: clave donde se guarda la contraseña en secrets/env
    - session_flag: nombre del flag en session_state
    """
    expected = _expected_password(password_key)

    if session_flag not in st.session_state:
        st.session_state[session_flag] = False

    with st.sidebar:
        st.subheader("Acceso")
        if not expected:
            st.info(
                f"Configura la contraseña en st.secrets['{password_key}'] "
                f"o en la variable de entorno {password_key}."
            )
        pwd = st.text_input("Contraseña", type="password", key="__login_pwd")
        c1, c2 = st.columns(2)
        login_clicked = c1.button("Entrar", use_container_width=True)
        logout_clicked = c2.button("Salir", use_container_width=True) if st.session_state[session_flag] else False

    if logout_clicked:
        st.session_state[session_flag] = False
        st.session_state["__login_pwd"] = ""
        st.rerun()

    # Sesión ya válida
    if st.session_state[session_flag]:
        return True

    # Validación cuando se pulsa Entrar
    if login_clicked:
        if expected and hmac.compare_digest(pwd or "", expected):
            st.session_state[session_flag] = True
            st.session_state["__login_pwd"] = ""
            st.success("✅ Acceso concedido.")
            st.rerun()
        else:
            st.error("❌ Contraseña incorrecta.")

    # Aún sin acceso
    st.write("🔒 La aplicación está protegida. Introduce la contraseña para continuar.")
    return False
