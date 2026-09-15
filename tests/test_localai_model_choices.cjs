const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const window = {};
vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../static/js/localai-models.js'), 'utf8'), {
    window, document: {createElement: () => ({})}
});
const model = 'Qwen/Custom Voice:EN';
const encoded = 'localai_tts::' + encodeURIComponent(model);
assert.equal(window.localAIEngineChoice(encoded).model, model);
assert.equal(window.localAIEngineChoice(encoded).engine, 'localai_tts');
assert.equal(window.localAIEngineChoice('index_tts').engine, 'index_tts');
const select = {value: 'kokoro', options: [], appendChild(option) {this.options.push(option);}};
window.appendLocalAIModelOptions(select, [{model_id: model}]);
window.appendLocalAIModelOptions(select, [{model_id: model}]);
assert.equal(select.options.length, 1);
assert.equal(select.value, 'kokoro');
assert.equal(select.options[0].textContent, `LocalAI: ${model}`);
const request = {engine: encoded, voice: {voice: 'Ryan', extra: {delivery_instruction: 'Calm'}}};
window.applyLocalAIModelToRegenRequest(request);
assert.equal(request.engine, 'localai_tts');
assert.equal(request.voice.voice, 'Ryan');
assert.equal(request.voice.extra.localai_tts_model, model);
assert.equal(request.voice.extra.delivery_instruction, 'Calm');
const other = {engine: 'breeze_tts_2'};
window.applyLocalAIModelToRegenRequest(other);
assert.deepEqual(other, {engine: 'breeze_tts_2'});
console.log('LocalAI model choice tests passed');
