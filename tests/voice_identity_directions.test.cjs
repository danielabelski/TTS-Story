const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

function context(body) {
    const ctx = vm.createContext({ console, document: {
        addEventListener() {}, getElementById: id => id === 'chunk-review-modal-body' ? body : null,
    }, window: {} });
    vm.runInContext(fs.readFileSync(path.join(__dirname, '../static/js/library.js'), 'utf8'), ctx);
    return ctx;
}

test('expanded Library speaker markup contains escaped persistent profile fields', () => {
    const body = { innerHTML: '', dataset: {} };
    const ctx = context(body);
    for (const name of ['wireChunkReviewEvents', 'wireChapterRebuildEvents',
        'wireChapterReviewRebuildEvent', 'wireBatchRebuildEvents']) ctx[name] = () => {};
    ctx.renderChunkReviewModal({ job_id: 'job', engine: 'breeze_tts_2', chunks: [{
        id: 'c1', speaker: 'barnaby-male', text: 'Hello.', voice_assignment: { extra: {
            speaker_profile: { description: 'A pompous creature.', voice: 'High-pitched, squeaky.',
                voice_design_prompt: '</textarea><script>unsafe()</script>' },
        } },
    }] });
    assert(body.innerHTML.includes('A pompous creature.'));
    assert(body.innerHTML.includes('High-pitched, squeaky.'));
    assert(body.innerHTML.includes('&lt;/textarea&gt;&lt;script&gt;'));
    assert(!body.innerHTML.includes('<script>unsafe()'));
    assert(body.innerHTML.includes('Save Speaker Properties'));
});

test('save speaker properties restores production then saves without synthesis', async () => {
    let click;
    const status = { textContent: '' };
    const fields = {
        '.speaker-profile-description': { value: 'A pompous creature.' },
        '.speaker-profile-voice': { value: 'High-pitched, squeaky.' },
        '.speaker-profile-design': { value: 'MALE VOICE. Squeaky.' },
        '.speaker-profile-save-status': status,
    };
    const button = { dataset: { speaker: 'barnaby-male' }, disabled: false,
        closest: () => ({ querySelector: key => fields[key] }),
        addEventListener: (event, handler) => { click = handler; },
    };
    const body = { querySelector: () => null,
        querySelectorAll: selector => selector === '.save-library-speaker-profile' ? [button] : [],
    };
    const ctx = context(body);
    const requests = [];
    ctx.populateLibraryVoiceSelects = async () => {};
    ctx.requestLibraryReviewRestore = async id => { requests.push(['restore', id]); };
    ctx.fetch = async (url, options) => {
        const payload = JSON.parse(options.body);
        requests.push([url, payload]);
        assert.equal(button.disabled, true);
        return { ok: true, json: async () => ({ success: true, profile: payload.profile }) };
    };
    vm.runInContext('chunkReviewModalData = {chunks: [{speaker:"barnaby-male", voice_assignment:{extra:{prompt_text:"Keep transcript"}}}]};', ctx);
    ctx.wireChunkReviewEvents('job', [], 'breeze_tts_2');
    await click();
    assert.equal(requests.length, 2);
    assert.deepEqual(requests[0], ['restore', 'job']);
    assert.equal(requests[1][0], '/api/jobs/job/review/speaker-profile');
    assert.equal(requests[1][1].profile.voice, 'High-pitched, squeaky.');
    assert.equal(button.disabled, false);
    assert.match(status.textContent, /Saved/);
    assert.equal(vm.runInContext('chunkReviewModalData.chunks[0].voice_assignment.extra.prompt_text', ctx), 'Keep transcript');
    assert.equal(vm.runInContext('chunkReviewModalData.chunks[0].voice_assignment.extra.speaker_profile.voice', ctx), 'High-pitched, squeaky.');
});

test('main submission resolves stored profile keys back to speaker assignment IDs', () => {
    const source = fs.readFileSync(path.join(__dirname, '../static/js/main.js'), 'utf8');
    const start = source.indexOf('speaker_profiles: Object.fromEntries(Object.keys(voiceAssignments)');
    const end = source.indexOf('review_mode:', start);
    const expression = source.slice(start, end).trim().replace(/,$/, '');
    const ctx = vm.createContext({ voiceAssignments: { 'barnaby-male': {}, narrator: {} },
        findSpeakerProfile: name => ({ profile: name === 'barnaby-male' ? { voice: 'Squeaky.' } : null }),
    });
    const result = vm.runInContext(`({${expression}})`, ctx);
    assert.deepEqual(Object.keys(result.speaker_profiles), ['barnaby-male']);
    assert.equal(result.speaker_profiles['barnaby-male'].voice, 'Squeaky.');
});
