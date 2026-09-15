// UI-only engine identities: the server still receives localai_tts + a model.
window.localAIEngineChoice = value => {
    const raw = String(value || '');
    const prefix = 'localai_tts::';
    return raw.startsWith(prefix)
        ? { engine: 'localai_tts', model: decodeURIComponent(raw.slice(prefix.length)) }
        : { engine: raw, model: '' };
};
window.appendLocalAIModelOptions = (select, models = window.localAIModelCatalog || []) => {
    if (!select) return;
    const selected = select.value;
    for (const model of models) {
        const id = model.model_id;
        if (!id) continue;
        const value = `localai_tts::${encodeURIComponent(id)}`;
        if (Array.from(select.options).some(option => option.value === value)) continue;
        const option = document.createElement('option');
        option.value = value;
        option.textContent = `LocalAI: ${model.name || id}`;
        select.appendChild(option);
    }
    if (selected) select.value = selected;
};
window.applyLocalAIModelToRegenRequest = request => {
    const choice = window.localAIEngineChoice(request.engine);
    if (choice.model) {
        request.engine = choice.engine;
        request.voice = { ...(request.voice || {}), extra: {
            ...(request.voice?.extra || {}), localai_tts_model: choice.model
        }};
    }
};
