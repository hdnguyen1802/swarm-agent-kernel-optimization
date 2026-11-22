'use strict';

const modelChoices = document.querySelectorAll('input[name="model"]');
const modelParts = [...document.querySelectorAll('[data-part]')];
const workflowButtons = [...document.querySelectorAll('[data-phase]')];
const workflowSteps = document.querySelector('.workflow-steps');
const finalModelButton = document.getElementById('final-model');
const targetContent = document.getElementById('target-content');
const previousButton = document.getElementById('previous-target');
const nextButton = document.getElementById('next-target');
const playButton = document.getElementById('play-replay');
const chartRows = document.querySelectorAll('[data-series]');
const raceButton = document.getElementById('play-race');
const raceStatus = document.getElementById('race-status');
const lanes = [...document.querySelectorAll('[data-racer]')].map(panel => ({
    panel,
    key: panel.dataset.racer,
    name: panel.querySelector('h3').textContent,
    progress: panel.querySelector('progress'),
    tokens: panel.querySelector('.lane-tokens'),
    time: panel.querySelector('.lane-time'),
    result: panel.querySelector('.lane-result'),
    output: panel.querySelector('.lane-output'),
}));
const maximumLatency = Math.ceil(Math.max(...Object.values(recordedRuns)
    .flatMap(run => Object.values(run.timing))) / 5) * 5;

let modelKey = document.querySelector('input[name="model"]:checked').value;
let run = recordedRuns[modelKey];
let targetIndex = 0;
let frameIndex = 0;
let frames = [];
let replayTimer = null;
let raceFrame = null;
let raceStart = 0;
let raceElapsed = 0;
let raceComplete = false;

function setText(id, value) {
    document.getElementById(id).textContent = value;
}

function ratio(value) {
    return `${value.toFixed(3)}×`;
}

function duration(ms) {
    return `${(ms / 1000).toFixed(3)} s`;
}

function resetRace() {
    if (raceFrame !== null) {
        window.cancelAnimationFrame(raceFrame);
    }
    raceFrame = null;
    raceElapsed = 0;
    raceComplete = false;
    const count = recordedAnswers.models[modelKey].tokenEnds.length;
    setText('race-model', run.name);
    setText('race-prompt-text', recordedAnswers.prompt);
    raceStatus.textContent = 'Play to replay all three answers side by side.';
    raceButton.textContent = 'Play race';
    for (const lane of lanes) {
        delete lane.panel.dataset.winner;
        lane.progress.max = count;
        lane.progress.value = 0;
        lane.tokens.textContent = `0 / ${count} tokens`;
        lane.time.textContent = '0.000 s';
        lane.result.textContent = '';
        lane.output.textContent = '';
    }
}

function tickRace(now) {
    const answer = recordedAnswers.models[modelKey];
    const count = answer.tokenEnds.length;
    raceElapsed = Math.max(0, now - raceStart);
    const winner = lanes.reduce((fastest, lane) => run.timing[lane.key] < run.timing[fastest.key] ? lane : fastest);
    const winnerTime = run.timing[winner.key] * (count - 1);
    let finished = 0;
    for (const lane of lanes) {
        const msPerToken = run.timing[lane.key];
        const total = msPerToken * (count - 1);
        const done = raceElapsed >= total;
        const emitted = done ? count : 1 + Math.floor(raceElapsed / msPerToken);
        if (lane.progress.value !== emitted) {
            lane.progress.value = emitted;
            lane.tokens.textContent = `${emitted} / ${count} tokens`;
            const follow = lane.output.scrollHeight - lane.output.scrollTop - lane.output.clientHeight < 4;
            lane.output.textContent = answer.text.slice(0, answer.tokenEnds[emitted - 1]);
            if (follow) {
                lane.output.scrollTop = lane.output.scrollHeight;
            }
        }
        lane.time.textContent = duration(Math.min(raceElapsed, total));
        if (done) {
            finished += 1;
            lane.panel.dataset.winner = String(lane === winner);
            lane.result.textContent = lane === winner ? 'First to finish' : 'Finished';
        }
    }
    const status = finished === lanes.length
        ? `${winner.name} finished first at ${duration(winnerTime)}. All ${count} output tokens matched.`
        : raceElapsed >= winnerTime
            ? `${winner.name} finished first at ${duration(winnerTime)}.`
            : 'Replaying the measured decode speeds.';
    if (raceStatus.textContent !== status) {
        raceStatus.textContent = status;
    }
    raceComplete = finished === lanes.length;
    if (raceComplete) {
        raceFrame = null;
        raceButton.textContent = 'Replay race';
    } else {
        raceFrame = window.requestAnimationFrame(tickRace);
    }
}

function playRace(restart = false) {
    if (restart || raceComplete) {
        resetRace();
    }
    raceStart = performance.now() - raceElapsed;
    raceButton.textContent = 'Pause race';
    raceFrame = window.requestAnimationFrame(tickRace);
}

const operationRoles = {
    'Gate + up projections': 'Expand and filter token features in the feed-forward block.',
    'Down projection': 'Bring the feed-forward output back to the model’s original width.',
    'Query projection': 'Turn token features into a query for attention.',
    'Output projection': 'Mix the results of the attention heads.',
    'Query + output projections': 'Make queries for attention and mix the results of its heads. The run grouped these two projections.',
    'Key + value projections': 'Make the keys and values that attention stores for each token.',
    'Output head': 'Turn the last layer’s output into scores for the next token.',
};

function element(tag, text, className = '') {
    const node = document.createElement(tag);
    node.textContent = text;
    node.className = className;
    return node;
}

function compareKernels(target, candidate) {
    const comparison = element('div', '', 'kernel-comparison');
    const original = element('div', '', 'original-kernel');
    const description = target.operator === 'aten::addmm'
        ? 'Standard GPU matrix multiply with bias'
        : 'Standard GPU matrix multiply';
    const baseline = element('p', `${description} · `);
    baseline.append(element('code', target.operator));
    original.append(element('span', 'Original · torch.compile', 'comparison-label'), baseline);
    comparison.append(original);
    const optimized = element('div', '', 'triton-kernel');
    optimized.append(element('span', 'SAKO candidate · Triton', 'comparison-label'),
        element('strong', candidate.change), element('p', candidate.summary, 'candidate-summary'));
    comparison.append(optimized);
    return comparison;
}

function metric(label, value) {
    const pair = element('div', '');
    pair.append(element('dt', label), element('dd', value));
    return pair;
}

function targetFrames(target) {
    if (!target) {
        return [];
    }
    if (!target.batches.length) {
        return [{phase: 'pick'}, {phase: 'kernel'}, {phase: 'decide'}];
    }
    const rounds = target.batches.flatMap((batch, index) =>
        ['model', 'decide'].map(phase => ({phase, batch, index})));
    return [{phase: 'pick'}, {phase: 'kernel'}, ...rounds];
}

const modelChecks = {
    passed: 'Passed model checks',
    correctness: 'Failed model correctness',
    compile: 'Failed model compilation',
    speed: 'Slower full model',
    unstable: 'Unstable decode timing',
    gate: 'Did not meet the full model requirements',
};

function showParallelSearch(target, detail) {
    if (!target.writers.length) {
        detail.append(element('p', 'This target had no writer jobs or candidate tests in the saved run.'));
        return;
    }
    const count = target.writers.length;
    detail.append(element('p', `${count} ${count === 1 ? 'writer was active' : 'writers were active at once'} at this target’s peak.`));
    const writers = element('ul', '', 'parallel-writers');
    writers.setAttribute('aria-label', 'Writer roles at peak concurrency');
    target.writers.forEach(role => writers.append(element('li', role)));
    detail.append(writers, element('p', 'Candidates can run as soon as a GPU is free, while other writers keep working.', 'target-note'));
    const tests = element('dl', '', 'parallel-gpus');
    target.gpuTests.forEach((count, gpu) => tests.append(metric(`GPU ${gpu}`, `${count} saved kernel ${count === 1 ? 'test' : 'tests'}`)));
    detail.append(tests);
}

function finalistCard(trial) {
    const card = element('div', '', 'finalist-card');
    card.dataset.gpu = trial.gpu;
    const header = element('div', '', 'finalist-header');
    header.append(element('strong', `GPU ${trial.gpu}`), element('span', `Rank ${trial.rank}`));
    const candidate = [run.rejected, ...run.selected].find(item => item.id === trial.id);
    const checkClass = trial.outcome === 'passed' ? 'target-used' : 'target-warning';
    card.append(header, element('p', candidate?.change ?? `${trial.specialty} writer’s kernel`, 'finalist-name'),
        element('p', modelChecks[trial.outcome], checkClass));
    if (trial.localRatio !== null) {
        card.append(element('p', `${ratio(trial.localRatio)} local kernel speedup`, 'target-note'));
    }
    if (['passed', 'speed'].includes(trial.outcome) && trial.modelRatio !== null) {
        card.append(element('p', `${ratio(trial.modelRatio)} decode speed vs the previous model`, 'target-note'));
    }
    return card;
}

function showFinalistDecision(target, nextBatch, detail) {
    if (nextBatch) {
        detail.append(element('p', 'Neither finalist passed the full model test.', 'target-warning'),
            element('p', `SAKO goes back to test ranks ${nextBatch.map(trial => trial.rank).join(' and ')}.`, 'target-note'));
        return;
    }
    if (target.visitWinner) {
        const winner = target.batches.flat().find(trial => trial.id === target.visitWinner);
        detail.append(element('p', `SAKO kept rank ${winner.rank} from this visit.`, 'target-used'));
    } else {
        const reason = target.batches.length ? 'No finalist passed the full model requirements.' : 'No candidate reached the full model test.';
        detail.append(element('p', 'SAKO kept the previous model at this visit.', 'target-retained'), element('p', reason, 'target-note'));
    }
    if (target.skippedRanks.length) {
        detail.append(element('p', `SAKO skipped ranks ${target.skippedRanks.join(' and ')} after a finalist passed.`, 'target-note'));
    }
    const final = run.selected.find(candidate => candidate.operation === target.operation);
    if (final && final.id !== target.visitWinner) {
        detail.append(element('p', `The final model uses a kernel that passed a later check. ${final.change}.`, 'target-note'));
    }
}

function showTargetStep(target, frame) {
    const {phase, batch, index} = frame;
    const label = batch ? `Batch ${index + 1} of ${target.batches.length}` : `Target ${targetIndex + 1} of ${run.targets.length}`;
    const title = element('div', '', 'target-heading');
    title.append(element('p', label, 'target-step'), element('h3', target.operation));
    const detail = element('div', '', 'target-detail');

    if (phase === 'pick') {
        const explanation = targetIndex === 0
            ? 'This target used the most GPU time among the targets in the profile.'
            : 'SAKO included this target in its search after profiling decode.';
        const profileTime = target.profileMs.toLocaleString('en-US', {maximumFractionDigits: 1});
        const metrics = element('dl', '', 'target-metrics');
        metrics.append(metric('GPU time over 127 decode steps', `${profileTime} ms`));
        detail.append(element('p', explanation), element('p', operationRoles[target.operation], 'target-note'), metrics);
    } else if (phase === 'kernel') {
        showParallelSearch(target, detail);
    } else if (phase === 'model') {
        const intro = batch.length === 2 ? 'The two GPUs tested these finalists in parallel.' : 'One finalist remained in this batch.';
        const cards = element('div', '', 'finalist-pair');
        cards.append(...batch.map(finalistCard));
        detail.append(element('p', intro), cards);
    } else {
        showFinalistDecision(target, frames[frameIndex + 1]?.batch, detail);
    }
    targetContent.replaceChildren(title, detail);
}

function renderReplay() {
    const target = run.targets[targetIndex];
    const frame = frames[frameIndex];
    const keptOperations = new Set(run.selected.map(candidate => candidate.operation));
    for (const button of modelParts) {
        const part = button.dataset.part;
        const active = target?.parts.includes(part) ?? false;
        const kept = !target && run.targets.some(item => keptOperations.has(item.operation) && item.parts.includes(part));
        button.dataset.state = active ? 'current' : kept ? 'kept' : 'idle';
        button.setAttribute('aria-pressed', String(active));
    }
    for (const button of workflowButtons) {
        const phase = button.dataset.phase;
        if (frame?.phase === phase) {
            button.setAttribute('aria-current', 'step');
        } else {
            button.removeAttribute('aria-current');
        }
        button.disabled = !frames.some(item => item.phase === phase);
    }
    const writerCount = target?.writers.length ?? run.peakWriters;
    setText('writer-overlap', writerCount ? `${writerCount} ${writerCount === 1 ? 'writer' : 'writers'} at the observed peak` : 'No writer jobs for this target');
    setText('model-batch-label', frame?.batch
        ? frame.batch.map(trial => `Rank ${trial.rank} on GPU ${trial.gpu}`).join(' · ')
        : 'One full model trial per GPU');
    const retrying = frame?.phase === 'decide' && frames[frameIndex + 1]?.phase === 'model';
    workflowSteps.classList.toggle('returning', retrying);
    const loop = document.getElementById('workflow-return');
    loop.classList.toggle('active', retrying);
    setText('workflow-return-label', retrying ? 'The first pair failed. Test the next finalists.' : 'If neither finalist passes, test the next pair when available.');
    setText('map-note', target ? 'The highlighted part is the current target.' : 'Highlights show the parts that use SAKO kernels.');
    finalModelButton.disabled = !target;
    previousButton.disabled = targetIndex === 0 && frameIndex === 0;
    nextButton.disabled = !target;
    if (frameIndex < frames.length - 1) {
        nextButton.textContent = retrying ? 'Test next finalists' : 'Next step';
    } else {
        nextButton.textContent = targetIndex < run.targets.length - 1 ? 'Next target' : 'Final result';
    }

    if (!target) {
        const {compiled, optimized} = run.timing;
        const metrics = element('dl', '', 'target-metrics');
        metrics.append(metric('torch.compile', `${compiled.toFixed(3)} ms/token`),
            metric('SAKO', `${optimized.toFixed(3)} ms/token`));
        const title = element('div', '', 'target-heading');
        title.append(element('p', 'Final model', 'target-step'), element('h3', `${ratio(compiled / optimized)} speedup vs torch.compile`));
        const detail = element('div', '', 'target-detail');
        detail.append(element('p', `SAKO kept ${run.selected.length} kernels after testing them in the complete model.`), metrics);
        targetContent.replaceChildren(title, detail);
    } else {
        showTargetStep(target, frame);
    }
}

function candidateRow(candidate, used) {
    const row = document.createElement('tr');
    const change = row.insertCell();
    const speedup = row.insertCell();
    const outcome = row.insertCell();
    const target = run.targets.find(item => item.operation === candidate.operation);
    const name = element('div', candidate.operation, 'candidate-target');
    name.append(element('code', candidate.id));
    change.append(name, compareKernels(target, candidate));
    if (!used) {
        change.append(element('p', candidate.reason, 'candidate-reason'));
    }
    speedup.textContent = ratio(candidate.speedup);
    outcome.append(element('span', used ? 'Used' : 'Rejected', used ? 'decision used' : 'decision rejected'));
    return row;
}

function renderModel() {
    setText('checkpoint', run.checkpoint);
    setText('map-model', run.checkpoint);
    setText('layer-count', `${run.layers} layers`);
    for (const row of chartRows) {
        const ms = run.timing[row.dataset.series];
        row.querySelector('.chart-value').textContent = ms.toFixed(3);
        row.style.setProperty('--bar-width', `${100 * ms / maximumLatency}%`);
    }
    setText('speedup', ratio(run.timing.pytorch / run.timing.optimized));
    setText('throughput', run.throughput.toFixed(2));
    setText('table-summary', `${run.name} used ${run.selected.length} kernels in the final model. This table also shows one rejection.`);
    document.getElementById('candidate-rows').replaceChildren(
        ...run.selected.map(candidate => candidateRow(candidate, true)),
        candidateRow(run.rejected, false),
    );
    resetRace();
    selectTarget(0);
}

function pauseReplay() {
    if (replayTimer !== null) {
        window.clearInterval(replayTimer);
        replayTimer = null;
    }
    const finished = frameIndex === frames.length - 1;
    setText('play-label', finished ? 'Replay this target' : 'Play this target');
}

function playReplay() {
    if (targetIndex === run.targets.length) {
        selectTarget(0);
    } else if (frameIndex === frames.length - 1) {
        frameIndex = 0;
        renderReplay();
    }
    setText('play-label', 'Pause walkthrough');
    replayTimer = window.setInterval(() => {
        frameIndex += 1;
        renderReplay();
        if (frameIndex === frames.length - 1) {
            pauseReplay();
        }
    }, 3000);
}

for (const choice of modelChoices) {
    choice.addEventListener('change', () => {
        modelKey = choice.value;
        run = recordedRuns[modelKey];
        renderModel();
    });
}

function selectTarget(index, lastStep = false) {
    targetIndex = index;
    frames = targetFrames(run.targets[targetIndex]);
    frameIndex = lastStep && frames.length ? frames.length - 1 : 0;
    pauseReplay();
    renderReplay();
}

function goToStep(index) {
    frameIndex = index;
    pauseReplay();
    renderReplay();
}

for (const button of modelParts) {
    button.addEventListener('click', () => {
        const index = run.targets.findIndex(target => target.parts.includes(button.dataset.part));
        if (index >= 0) {
            selectTarget(index);
        }
    });
}

for (const button of workflowButtons) {
    button.addEventListener('click', () => {
        const phase = button.dataset.phase;
        const trialIndex = frames[frameIndex].index ?? 0;
        const index = frames.findIndex(frame => frame.phase === phase && (frame.index ?? trialIndex) === trialIndex);
        if (index >= 0) {
            goToStep(index);
        }
    });
}

previousButton.addEventListener('click', () => {
    if (frameIndex > 0) {
        goToStep(frameIndex - 1);
    } else {
        selectTarget(targetIndex - 1, true);
    }
});
nextButton.addEventListener('click', () => {
    if (frameIndex < frames.length - 1) {
        goToStep(frameIndex + 1);
    } else {
        selectTarget(targetIndex + 1);
    }
});
finalModelButton.addEventListener('click', () => selectTarget(run.targets.length));

playButton.addEventListener('click', () => {
    if (replayTimer === null) {
        playReplay();
    } else {
        pauseReplay();
    }
});

raceButton.addEventListener('click', () => {
    if (raceFrame === null) {
        playRace();
    } else {
        window.cancelAnimationFrame(raceFrame);
        raceFrame = null;
        raceButton.textContent = 'Resume race';
        raceStatus.textContent = 'Race paused.';
    }
});

document.getElementById('start-race').addEventListener('click', () => {
    document.getElementById('race-title').focus({preventScroll: true});
    document.getElementById('output-race').scrollIntoView();
    playRace(true);
});

const candidateDetails = document.getElementById('candidates');
function openLinkedCandidates() {
    if (location.hash === '#candidates') {
        candidateDetails.open = true;
    }
}
document.querySelector('a[href="#candidates"]').addEventListener('click', () => {
    candidateDetails.open = true;
});
window.addEventListener('hashchange', openLinkedCandidates);
openLinkedCandidates();
renderModel();
