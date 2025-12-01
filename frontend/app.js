const API_BASE = "http://localhost:8000";

// Navigation
const navItems = document.querySelectorAll(".nav-item");
const ingestView = document.getElementById("view-ingest");
const assistantView = document.getElementById("view-assistant");

navItems.forEach((btn) => {
    btn.addEventListener("click", () => {
        navItems.forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");

        const view = btn.dataset.view;
        if (view === "ingest") {
            ingestView.classList.add("view-active");
            assistantView.classList.remove("view-active");
        } else if (view === "assistant") {
            assistantView.classList.add("view-active");
            ingestView.classList.remove("view-active");
        }
    });
});

// Ingest related
const fileInput = document.getElementById("fileInput");
const ingestBtn = document.getElementById("ingestBtn");
const ingestStatus = document.getElementById("ingestStatus");

// Assistant related
const questionInput = document.getElementById("questionInput");
const topKInput = document.getElementById("topKInput");
const modeSelect = document.getElementById("modeSelect");
const askBtn = document.getElementById("askBtn");
const queryStatus = document.getElementById("queryStatus");
const modelInfo = document.getElementById("modelInfo");

const answerBox = document.getElementById("answerBox");
const sourcesList = document.getElementById("sourcesList");
const togglePromptBtn = document.getElementById("togglePromptBtn");
const promptBox = document.getElementById("promptBox");

let lastRawPrompt = "";
let lastModelName = "";

function setStatus(el, message, type) {
    el.textContent = message;
    el.classList.remove("ok", "error");
    if (type) el.classList.add(type);
}

async function ingestFiles() {
    if (!fileInput.files.length) {
        setStatus(ingestStatus, "Please choose at least one .txt file.", "error");
        return;
    }

    const formData = new FormData();
    for (const file of fileInput.files) {
        formData.append("files", file);
    }

    ingestBtn.disabled = true;
    setStatus(ingestStatus, "Ingesting and indexing locally…", "");

    try {
        const resp = await fetch(`${API_BASE}/api/ingest`, {
            method: "POST",
            body: formData,
        });

        if (!resp.ok) {
            const text = await resp.text();
            throw new Error(text || `HTTP ${resp.status}`);
        }

        const data = await resp.json();
        setStatus(
            ingestStatus,
            `Ingested successfully. Created ${data.chunks} semantic chunks.`,
            "ok"
        );
    } catch (err) {
        console.error(err);
        setStatus(
            ingestStatus,
            `Failed to ingest: ${err.message || err}`,
            "error"
        );
    } finally {
        ingestBtn.disabled = false;
    }
}

async function askQuestion() {
    const question = questionInput.value.trim();
    const topK = parseInt(topKInput.value, 10) || 4;
    const mode = modeSelect.value || "standard";

    if (!question) {
        setStatus(queryStatus, "Please enter a question.", "error");
        return;
    }

    askBtn.disabled = true;
    setStatus(queryStatus, "Running local RAG + LLM…", "");
    answerBox.textContent = "";
    sourcesList.innerHTML = "";
    promptBox.textContent = "";
    lastRawPrompt = "";
    promptBox.classList.add("hidden");

    try {
        const resp = await fetch(`${API_BASE}/api/query`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ question, top_k: topK, mode }),
        });

        if (!resp.ok) {
            const text = await resp.text();
            throw new Error(text || `HTTP ${resp.status}`);
        }

        const data = await resp.json();
        answerBox.textContent = data.answer || "(no answer returned)";

        lastModelName = data.model_name || "";
        const modeLabel = data.mode || mode;
        if (lastModelName) {
            modelInfo.textContent = `Model: ${lastModelName} · Mode: ${modeLabel}`;
        } else {
            modelInfo.textContent = `Mode: ${modeLabel}`;
        }

        // Render chunks
        sourcesList.innerHTML = "";
        (data.chunks || []).forEach((chunk) => {
            const item = document.createElement("article");
            item.className = "source-item";

            const meta = document.createElement("div");
            meta.className = "source-meta";
            const left = document.createElement("span");
            left.textContent = `Source ${chunk.rank}`;
            const right = document.createElement("span");
            right.textContent = `distance: ${chunk.distance.toFixed(4)}`;
            meta.appendChild(left);
            meta.appendChild(right);

            const text = document.createElement("div");
            text.className = "source-text";
            text.textContent = chunk.text;

            item.appendChild(meta);
            item.appendChild(text);
            sourcesList.appendChild(item);
        });

        lastRawPrompt = data.raw_prompt || "";
        if (lastRawPrompt) {
            promptBox.textContent = lastRawPrompt;
        }

        setStatus(queryStatus, "Done.", "ok");
    } catch (err) {
        console.error(err);
        setStatus(
            queryStatus,
            `Failed to query: ${err.message || err}`,
            "error"
        );
    } finally {
        askBtn.disabled = false;
    }
}

ingestBtn.addEventListener("click", ingestFiles);
askBtn.addEventListener("click", askQuestion);

togglePromptBtn.addEventListener("click", () => {
    if (!lastRawPrompt) {
        promptBox.textContent = "(No prompt available yet. Run a query first.)";
    }
    promptBox.classList.toggle("hidden");
});
