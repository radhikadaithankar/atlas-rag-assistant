"""One Streamlit page. Every call goes through the HTTP API."""

from __future__ import annotations

import streamlit as st

from atlas.ui.client import (
    AtlasClient,
    AtlasClientError,
    ask_error_text,
    chunk_label,
    format_place,
)

BOUNDARY = "One local collection, no user isolation. Do not expose this UI with private documents."
CAPTION = (
    "Keyword search does not need a key. Ask needs OPENAI_API_KEY on the API, "
    "including keyword mode, because it generates an answer."
)


def main() -> None:
    st.title("Atlas")
    st.info(BOUNDARY)
    st.caption(CAPTION)
    client = AtlasClient()
    try:
        health = client.health()
        st.write(f"API health: {health['status']}")
    except AtlasClientError:
        st.write("API health: unreachable")
    mode = st.selectbox("Mode", ["keyword", "vector", "hybrid"], index=0)
    rerank = st.checkbox("Rerank", value=False)
    upload = st.file_uploader("Document", type=["pdf", "txt", "md"])
    if st.button("Upload"):
        if upload is None:
            st.error("Choose a file first.")
        else:
            try:
                stored = client.upload(upload.name, upload.getvalue())
                st.success(f"Stored {stored.filename}.")
            except AtlasClientError as exc:
                st.error(ask_error_text(exc.status, exc.message))
    _show_documents(client)
    question = st.text_input("Question")
    column_search, column_ask = st.columns(2)
    if column_search.button("Search") and question:
        _show_search(client, question, str(mode), bool(rerank))
    if column_ask.button("Ask") and question:
        _show_ask(client, question, str(mode), bool(rerank))


def _show_documents(client: AtlasClient) -> None:
    try:
        listing = client.list_documents()
    except AtlasClientError as exc:
        st.error(ask_error_text(exc.status, exc.message))
        return
    if not listing.items:
        st.write("No documents stored.")
        return
    for document in listing.items:
        label = f"{document.filename} ({chunk_label(document.chunk_count)})"
        st.write(label)
        if st.button("Delete", key=f"delete-{document.id}"):
            try:
                client.delete_document(str(document.id))
            except AtlasClientError as exc:
                st.error(ask_error_text(exc.status, exc.message))
            else:
                st.rerun()


def _show_search(client: AtlasClient, question: str, mode: str, rerank: bool) -> None:
    try:
        result = client.search(question, mode=mode, rerank=rerank)
    except AtlasClientError as exc:
        st.error(ask_error_text(exc.status, exc.message))
        return
    if not result.hits:
        st.write("No passages matched.")
        return
    for hit in result.hits:
        st.write(format_place(hit.filename, hit.page, hit.section))
        if hit.section:
            st.subheader(hit.section)
        st.write(hit.text)


def _show_ask(client: AtlasClient, question: str, mode: str, rerank: bool) -> None:
    try:
        result = client.ask(question, mode=mode, rerank=rerank)
    except AtlasClientError as exc:
        st.error(ask_error_text(exc.status, exc.message))
        return
    st.write(result.answer)
    st.write(f"Abstained {'yes' if result.abstained else 'no'}")
    if result.latency_ms is not None:
        st.write(f"Latency {int(result.latency_ms)} ms")
    if result.prompt_tokens is not None or result.completion_tokens is not None:
        st.caption(
            f"Tokens: prompt {result.prompt_tokens}, completion {result.completion_tokens}"
        )
    for citation in result.citations:
        st.write(format_place(citation.filename, citation.page, citation.section))
        st.write(citation.quote)


if __name__ == "__main__":
    main()
