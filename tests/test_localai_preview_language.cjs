const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('static/js/main.js', 'utf8');
const start = source.indexOf('function resolveLocalAILanguage(');
const end = source.indexOf('function createAssignment(', start);
let language = ' en ';
const context = {
    getAssignmentRows: () => [{dataset: {speaker: 'Bob'}, querySelector: () => ({value: language})}],
    runtimeSettings: {localai_tts_default_language: 'fr'},
};
vm.createContext(context);
vm.runInContext(source.slice(start, end), context);
assert.equal(context.resolveLocalAILanguage('Bob'), 'en');
language = 'en-US';
assert.equal(context.resolveLocalAILanguage('Bob'), 'en-US');
language = '';
assert.equal(context.resolveLocalAILanguage('Bob'), 'fr');
context.runtimeSettings.localai_tts_default_language = '';
assert.equal(context.resolveLocalAILanguage('Bob'), '');
assert.equal(context.resolveLocalAILanguage('unknown'), '');
const preview = source.slice(source.indexOf('async function handleFxPreview('), source.indexOf('async function handleFxPreview(') + 4500);
assert.match(preview, /isLocalAITtsEngine\(engineName\)\s*\? resolveLocalAILanguage\(speaker\)/);
assert.match(source, /const language = resolveLocalAILanguage\(speaker\);/);
console.log('LocalAI preview and job language regression checks passed.');
