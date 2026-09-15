const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../static/js/library.js'), 'utf8');
const start = source.indexOf('function resolveLocalAIFreeformSelection(');
const end = source.indexOf('\nasync function populateLibraryVoiceSelects', start);
const context = vm.createContext({});
vm.runInContext(source.slice(start, end), context);
const resolve = context.resolveLocalAIFreeformSelection;
for (const voice of ['Ryan', 'Dylan', 'Vivian', 'custom/voice.wav']) {
    const select = {
        value: 'discovered-voice',
        _localAIFields: {
            hidden: false,
            querySelector: selector => ({value: selector.endsWith('voice-id') ? ` ${voice} ` : ' en-US '})
        }
    };
    const result = resolve(select);
    assert.equal(result.voice, voice);
    assert.equal(result.language, 'en-US');
    assert.equal(result.cancelled, false);
    select._localAIFields.hidden = true;
    assert.equal(resolve(select), null);
}
assert.equal(resolve({value: 'Ryan'}), null);
// Both chunk and bulk population must update controls, including engine switches.
assert.equal(source.split("updateLocalAIManualFields(select, engineName.includes('localaitts'));").length - 1, 2);
console.log('LocalAI Library manual ID tests passed');
