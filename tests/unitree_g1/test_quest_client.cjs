// Run: node --test tests/unitree_g1/test_quest_client.cjs
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

function client(fetch) {
  const elements = new Map();
  const ctx = vm.createContext({
    document: {getElementById: id => {
      if (!elements.has(id)) elements.set(id, {value: '', disabled: false, addEventListener() {}});
      return elements.get(id);
    }},
    location: {hash: '#token=test'}, URLSearchParams,
    navigator: {}, isSecureContext: true, performance: {now: () => 100},
    AbortSignal, fetch, setInterval() {}, clearInterval() {}, console,
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../../scripts/unitree_g1/quest_web/client.js'), 'utf8'), ctx);
  vm.runInContext('episode = "test-episode"', ctx);
  return {ctx, run: code => vm.runInContext(code, ctx)};
}

test('end waits for in-flight batch and flushes final event before closing', async () => {
  const requests = [];
  let release;
  const {run} = client(async (url, options) => {
    const body = JSON.parse(options.body);
    requests.push({url, body});
    if (requests.length === 1) await new Promise(resolve => { release = resolve; });
    return {ok: true, json: async () => url.endsWith('/end') ? {frames: 0, valid_both_frames: 0} : {ack: body.records.at(-1).seq}};
  });
  run('event("session_start"); flush()');
  const done = run('finish()');
  release();
  await done;
  assert.deepEqual(requests.map(r => r.url), ['/api/batch', '/api/batch', '/api/end']);
  assert.equal(requests[0].body.records[0].seq, 0);
  assert.equal(requests[1].body.records[0].name, 'session_end');
  assert.equal(requests[1].body.records[0].seq, 1);
});

test('transport errors stop recording without silently retrying ambiguous batches', async () => {
  let calls = 0;
  const {run} = client(async () => { calls++; throw new Error('network lost'); });
  await run('event("session_start"); flush()');
  await run('event("more"); flush()');
  assert.equal(calls, 1);
  assert.equal(run('failed'), true);
});

test('queue overflow explicitly faults; source loss remains null', () => {
  const {run} = client(async () => {});
  assert.equal(run('pose(null)'), null);
  run('for(let i=0; i<241; i++) event("test")');
  assert.equal(run('failed'), true);
  assert.equal(run('pending.length'), 240);
});
