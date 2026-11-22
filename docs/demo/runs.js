// Selected measurements from the recorded GPU results in docs/validation.md.
// Candidate summaries describe the saved writer designs and Triton source.
// Targets follow the first visit to each operation in the saved run.
// Writer overlap comes from start and finish events. Finalist pairs come from the first visit.
const recordedRuns = {
    "mistral": {
        "name": "Mistral",
        "checkpoint": "Mistral-7B-Instruct-v0.3",
        "timing": {
            "pytorch": 29.913755,
            "compiled": 26.666864,
            "optimized": 21.547589
        },
        "throughput": 46.408905,
        "selected": [
            {
                "id": "0d466c0ee",
                "operation": "Gate + up projections",
                "speedup": 1.555174,
                "change": "Reuse input across rows",
                "summary": "Load each input block once and use it for several output rows during decode."
            },
            {
                "id": "87d7c1f15",
                "operation": "Down projection",
                "speedup": 1.204577,
                "change": "Keep input in cache",
                "summary": "Give the reused input higher cache priority. Skip bounds checks when the blocks fit exactly."
            },
            {
                "id": "bf149672a",
                "operation": "Query + output projections",
                "speedup": 1.199083,
                "change": "Sum each output row once",
                "summary": "Hold partial sums in GPU registers and add them together at the end of each row."
            },
            {
                "id": "e219f60ce",
                "operation": "Key + value projections",
                "speedup": 1.034029,
                "change": "Stream weights through cache",
                "summary": "Give the reused input higher cache priority than the weights."
            }
        ],
        "rejected": {
            "id": "fe28f0644",
            "operation": "Gate + up projections",
            "speedup": 1.519671,
            "reason": "The kernel passed its local test. The complete model failed correctness.",
            "modelRatio": null,
            "change": "Combine cache reuse and a final sum",
            "summary": "Give the input higher cache priority and add each row’s partial sums at the end."
        },
        "layers": 32,
        "targets": [
            {
                "operation": "Gate + up projections",
                "parts": [
                    "gate-up"
                ],
                "operator": "aten::mm",
                "profileMs": 1688.841,
                "writers": [
                    "Memory",
                    "Scheduling",
                    "Compute",
                    "Registers",
                    "Scheduling"
                ],
                "gpuTests": [
                    8,
                    7
                ],
                "batches": [
                    [
                        {
                            "id": "0d466c0ee",
                            "rank": 1,
                            "gpu": 0,
                            "specialty": "Memory",
                            "outcome": "correctness",
                            "localRatio": 1.555174,
                            "modelRatio": null
                        },
                        {
                            "id": "fe28f0644",
                            "rank": 2,
                            "gpu": 1,
                            "specialty": "Compute",
                            "outcome": "correctness",
                            "localRatio": 1.519671,
                            "modelRatio": null
                        }
                    ],
                    [
                        {
                            "id": "30042c104",
                            "rank": 3,
                            "gpu": 0,
                            "specialty": "Memory",
                            "outcome": "passed",
                            "localRatio": 1.51258,
                            "modelRatio": 1.148622
                        },
                        {
                            "id": "1d7706929",
                            "rank": 4,
                            "gpu": 1,
                            "specialty": "Memory",
                            "outcome": "passed",
                            "localRatio": 1.510704,
                            "modelRatio": 1.157549
                        }
                    ]
                ],
                "visitWinner": "1d7706929",
                "skippedRanks": []
            },
            {
                "operation": "Down projection",
                "parts": [
                    "down"
                ],
                "operator": "aten::mm",
                "profileMs": 680.669,
                "writers": [
                    "Memory",
                    "Scheduling",
                    "Compute",
                    "Registers",
                    "Fusion"
                ],
                "gpuTests": [
                    4,
                    4
                ],
                "batches": [
                    [
                        {
                            "id": "8ce761551",
                            "rank": 1,
                            "gpu": 0,
                            "specialty": "Compute",
                            "outcome": "passed",
                            "localRatio": 1.239354,
                            "modelRatio": 1.030502
                        },
                        {
                            "id": "87d7c1f15",
                            "rank": 2,
                            "gpu": 1,
                            "specialty": "Memory",
                            "outcome": "passed",
                            "localRatio": 1.204577,
                            "modelRatio": 1.038806
                        }
                    ]
                ],
                "visitWinner": "87d7c1f15",
                "skippedRanks": [
                    3,
                    4
                ]
            },
            {
                "operation": "Query + output projections",
                "parts": [
                    "query",
                    "attention-output"
                ],
                "operator": "aten::mm",
                "profileMs": 448.331,
                "writers": [
                    "Memory",
                    "Scheduling",
                    "Compute",
                    "Registers",
                    "Scheduling"
                ],
                "gpuTests": [
                    7,
                    3
                ],
                "batches": [
                    [
                        {
                            "id": "fd092719e",
                            "rank": 1,
                            "gpu": 0,
                            "specialty": "Memory",
                            "outcome": "correctness",
                            "localRatio": 1.199449,
                            "modelRatio": null
                        },
                        {
                            "id": "bf149672a",
                            "rank": 2,
                            "gpu": 1,
                            "specialty": "Compute",
                            "outcome": "gate",
                            "localRatio": 1.199083,
                            "modelRatio": 1.01297
                        }
                    ],
                    [
                        {
                            "id": "bd5ac04c6",
                            "rank": 3,
                            "gpu": 0,
                            "specialty": "Memory",
                            "outcome": "correctness",
                            "localRatio": 1.174549,
                            "modelRatio": null
                        },
                        {
                            "id": "9dda98388",
                            "rank": 4,
                            "gpu": 1,
                            "specialty": "Scheduling",
                            "outcome": "correctness",
                            "localRatio": 1.168623,
                            "modelRatio": null
                        }
                    ]
                ],
                "visitWinner": null,
                "skippedRanks": []
            },
            {
                "operation": "Key + value projections",
                "parts": [
                    "key-value"
                ],
                "operator": "aten::mm",
                "profileMs": 149.349,
                "writers": [
                    "Memory",
                    "Scheduling"
                ],
                "gpuTests": [
                    3,
                    1
                ],
                "batches": [
                    [
                        {
                            "id": "052231186",
                            "rank": 1,
                            "gpu": 0,
                            "specialty": "Scheduling",
                            "outcome": "compile",
                            "localRatio": 1.224996,
                            "modelRatio": null
                        },
                        {
                            "id": "19af38719",
                            "rank": 2,
                            "gpu": 1,
                            "specialty": "Compute",
                            "outcome": "correctness",
                            "localRatio": 1.179174,
                            "modelRatio": null
                        }
                    ],
                    [
                        {
                            "id": "e219f60ce",
                            "rank": 3,
                            "gpu": 0,
                            "specialty": "Memory",
                            "outcome": "gate",
                            "localRatio": 1.034029,
                            "modelRatio": 1.015995
                        }
                    ]
                ],
                "visitWinner": null,
                "skippedRanks": []
            },
            {
                "operation": "Output head",
                "parts": [
                    "output-head"
                ],
                "operator": "aten::mm",
                "profileMs": 51.766,
                "writers": [],
                "gpuTests": [
                    0,
                    0
                ],
                "batches": [],
                "visitWinner": null,
                "skippedRanks": []
            }
        ],
        "peakWriters": 5
    },
    "qwen": {
        "name": "Qwen",
        "checkpoint": "Qwen2.5-7B-Instruct",
        "timing": {
            "pytorch": 28.691278,
            "compiled": 25.621234,
            "optimized": 20.915339
        },
        "throughput": 47.8118,
        "selected": [
            {
                "id": "72a5c5ed4",
                "operation": "Gate + up projections",
                "speedup": 1.40593,
                "change": "Tune block size and cache reuse",
                "summary": "Test larger blocks. Give the reused input higher cache priority than the weights."
            },
            {
                "id": "00b2fbdf7",
                "operation": "Down projection",
                "speedup": 1.157251,
                "change": "Compute each row in one launch",
                "summary": "Multiply and sum the values for each output row in one GPU launch."
            },
            {
                "id": "36381c7cb",
                "operation": "Output head",
                "speedup": 1.291811,
                "change": "Read adjacent weights together",
                "summary": "Read blocks of adjacent weights and give the reused input higher cache priority."
            },
            {
                "id": "60694a285",
                "operation": "Key + value projections",
                "speedup": 1.477495,
                "change": "Share input across output rows",
                "summary": "Load each input block once for a group of output rows during decode."
            },
            {
                "id": "b94810808",
                "operation": "Query projection",
                "speedup": 1.143212,
                "change": "Use compact blocks for decode",
                "summary": "Read adjacent weights in blocks to limit the values held in registers."
            }
        ],
        "rejected": {
            "id": "de6deabb1",
            "operation": "Output projection",
            "speedup": 1.143859,
            "reason": "The complete model passed correctness. Decode was slower than the accepted model.",
            "modelRatio": 0.989117,
            "change": "Reuse input in a decode kernel",
            "summary": "Use the same input block for several output rows in a kernel for one token."
        },
        "layers": 28,
        "targets": [
            {
                "operation": "Gate + up projections",
                "parts": [
                    "gate-up"
                ],
                "operator": "aten::mm",
                "profileMs": 1604.146,
                "writers": [
                    "Memory",
                    "Scheduling",
                    "Compute",
                    "Registers",
                    "Fusion"
                ],
                "gpuTests": [
                    8,
                    8
                ],
                "batches": [
                    [
                        {
                            "id": "72a5c5ed4",
                            "rank": 1,
                            "gpu": 0,
                            "specialty": "Memory",
                            "outcome": "passed",
                            "localRatio": 1.40593,
                            "modelRatio": 1.148084
                        },
                        {
                            "id": "63d27234b",
                            "rank": 2,
                            "gpu": 1,
                            "specialty": "Memory",
                            "outcome": "passed",
                            "localRatio": 1.402142,
                            "modelRatio": 1.138813
                        }
                    ]
                ],
                "visitWinner": "72a5c5ed4",
                "skippedRanks": [
                    3,
                    4
                ]
            },
            {
                "operation": "Down projection",
                "parts": [
                    "down"
                ],
                "operator": "aten::mm",
                "profileMs": 673.301,
                "writers": [
                    "Memory",
                    "Scheduling",
                    "Compute",
                    "Registers",
                    "Fusion"
                ],
                "gpuTests": [
                    4,
                    3
                ],
                "batches": [
                    [
                        {
                            "id": "f241cbda7",
                            "rank": 1,
                            "gpu": 0,
                            "specialty": "Memory",
                            "outcome": "passed",
                            "localRatio": 1.165745,
                            "modelRatio": 1.010943
                        },
                        {
                            "id": "00b2fbdf7",
                            "rank": 2,
                            "gpu": 1,
                            "specialty": "Registers",
                            "outcome": "passed",
                            "localRatio": 1.157251,
                            "modelRatio": 1.02432
                        }
                    ]
                ],
                "visitWinner": "00b2fbdf7",
                "skippedRanks": [
                    3,
                    4
                ]
            },
            {
                "operation": "Output head",
                "parts": [
                    "output-head"
                ],
                "operator": "aten::mm",
                "profileMs": 202.985,
                "writers": [
                    "Memory",
                    "Scheduling",
                    "Compute",
                    "Registers"
                ],
                "gpuTests": [
                    2,
                    2
                ],
                "batches": [
                    [
                        {
                            "id": "36381c7cb",
                            "rank": 1,
                            "gpu": 0,
                            "specialty": "Memory",
                            "outcome": "passed",
                            "localRatio": 1.291811,
                            "modelRatio": 1.018304
                        },
                        {
                            "id": "ea8c22e15",
                            "rank": 2,
                            "gpu": 1,
                            "specialty": "Registers",
                            "outcome": "passed",
                            "localRatio": 1.27382,
                            "modelRatio": 1.011189
                        }
                    ]
                ],
                "visitWinner": "36381c7cb",
                "skippedRanks": [
                    3,
                    4
                ]
            },
            {
                "operation": "Output projection",
                "parts": [
                    "attention-output"
                ],
                "operator": "aten::mm",
                "profileMs": 151.432,
                "writers": [
                    "Memory",
                    "Scheduling",
                    "Compute",
                    "Registers"
                ],
                "gpuTests": [
                    1,
                    1
                ],
                "batches": [
                    [
                        {
                            "id": "de6deabb1",
                            "rank": 1,
                            "gpu": 0,
                            "specialty": "Memory",
                            "outcome": "speed",
                            "localRatio": 1.143859,
                            "modelRatio": 0.989117
                        },
                        {
                            "id": "5dd4b6a45",
                            "rank": 2,
                            "gpu": 1,
                            "specialty": "Scheduling",
                            "outcome": "speed",
                            "localRatio": 1.061682,
                            "modelRatio": 0.998709
                        }
                    ]
                ],
                "visitWinner": null,
                "skippedRanks": []
            },
            {
                "operation": "Key + value projections",
                "parts": [
                    "key-value"
                ],
                "operator": "aten::addmm",
                "profileMs": 142.228,
                "writers": [
                    "Memory",
                    "Scheduling",
                    "Compute",
                    "Registers",
                    "Fusion"
                ],
                "gpuTests": [
                    2,
                    0
                ],
                "batches": [
                    [
                        {
                            "id": "7ecd92824",
                            "rank": 1,
                            "gpu": 0,
                            "specialty": "Compute",
                            "outcome": "passed",
                            "localRatio": 2.298555,
                            "modelRatio": 1.011212
                        },
                        {
                            "id": "60694a285",
                            "rank": 2,
                            "gpu": 1,
                            "specialty": "Memory",
                            "outcome": "passed",
                            "localRatio": 1.477495,
                            "modelRatio": 1.013889
                        }
                    ]
                ],
                "visitWinner": "60694a285",
                "skippedRanks": []
            },
            {
                "operation": "Query projection",
                "parts": [
                    "query"
                ],
                "operator": "aten::addmm",
                "profileMs": 136.212,
                "writers": [
                    "Memory",
                    "Scheduling",
                    "Compute",
                    "Registers",
                    "Fusion"
                ],
                "gpuTests": [
                    2,
                    3
                ],
                "batches": [
                    [
                        {
                            "id": "b94810808",
                            "rank": 1,
                            "gpu": 0,
                            "specialty": "Registers",
                            "outcome": "passed",
                            "localRatio": 1.143212,
                            "modelRatio": 1.00499
                        },
                        {
                            "id": "5b160061a",
                            "rank": 2,
                            "gpu": 1,
                            "specialty": "Memory",
                            "outcome": "speed",
                            "localRatio": 1.112265,
                            "modelRatio": 0.998702
                        }
                    ]
                ],
                "visitWinner": "b94810808",
                "skippedRanks": [
                    3
                ]
            }
        ],
        "peakWriters": 5
    },
    "llama": {
        "name": "Llama",
        "checkpoint": "Llama-3.1-8B",
        "timing": {
            "pytorch": 30.631862,
            "compiled": 27.589473,
            "optimized": 22.837235
        },
        "throughput": 43.788138,
        "selected": [
            {
                "id": "eb8a1641c",
                "operation": "Gate + up projections",
                "speedup": 1.351545,
                "change": "Skip checks on full input blocks",
                "summary": "Skip bounds checks when the input fits the blocks exactly. Test wider blocks."
            },
            {
                "id": "2061b4fdb",
                "operation": "Down projection",
                "speedup": 1.167777,
                "change": "Use smaller blocks",
                "summary": "Process fewer output rows and weight values in each block to limit register use."
            },
            {
                "id": "4033c8066",
                "operation": "Query + output projections",
                "speedup": 1.255218,
                "change": "Use direct loads on full blocks",
                "summary": "Skip bounds checks for full input and output blocks. Test wider input blocks."
            },
            {
                "id": "a2edf5aad",
                "operation": "Output head",
                "speedup": 1.157671,
                "change": "Tune how threads share the work",
                "summary": "Test smaller groups of GPU threads and different block sizes."
            },
            {
                "id": "cab906273",
                "operation": "Key + value projections",
                "speedup": 1.136979,
                "change": "Split each row across GPU threads",
                "summary": "Compute parts of each output row in parallel and add the results in FP32."
            }
        ],
        "rejected": {
            "id": "cd9e89169",
            "operation": "Down projection",
            "speedup": 1.196453,
            "reason": "The complete model passed correctness. Its decode times varied too much to confirm a gain.",
            "modelRatio": 1.038036,
            "change": "Cache input and test wider blocks",
            "summary": "Give the input higher cache priority. Skip bounds checks on full input blocks."
        },
        "layers": 32,
        "targets": [
            {
                "operation": "Gate + up projections",
                "parts": [
                    "gate-up"
                ],
                "operator": "aten::mm",
                "profileMs": 1674.278,
                "writers": [
                    "Memory",
                    "Scheduling",
                    "Compute",
                    "Registers",
                    "Fusion"
                ],
                "gpuTests": [
                    9,
                    8
                ],
                "batches": [
                    [
                        {
                            "id": "eb8a1641c",
                            "rank": 1,
                            "gpu": 0,
                            "specialty": "Memory",
                            "outcome": "passed",
                            "localRatio": 1.351545,
                            "modelRatio": 1.128982
                        },
                        {
                            "id": "491e16ebe",
                            "rank": 2,
                            "gpu": 1,
                            "specialty": "Scheduling",
                            "outcome": "correctness",
                            "localRatio": 1.348794,
                            "modelRatio": null
                        }
                    ]
                ],
                "visitWinner": "eb8a1641c",
                "skippedRanks": [
                    3,
                    4
                ]
            },
            {
                "operation": "Down projection",
                "parts": [
                    "down"
                ],
                "operator": "aten::mm",
                "profileMs": 682.301,
                "writers": [
                    "Memory",
                    "Scheduling",
                    "Compute",
                    "Registers",
                    "Scheduling"
                ],
                "gpuTests": [
                    5,
                    3
                ],
                "batches": [
                    [
                        {
                            "id": "cd9e89169",
                            "rank": 1,
                            "gpu": 0,
                            "specialty": "Memory",
                            "outcome": "unstable",
                            "localRatio": 1.196453,
                            "modelRatio": 1.038036
                        },
                        {
                            "id": "2061b4fdb",
                            "rank": 2,
                            "gpu": 1,
                            "specialty": "Registers",
                            "outcome": "passed",
                            "localRatio": 1.167777,
                            "modelRatio": 1.035021
                        }
                    ]
                ],
                "visitWinner": "2061b4fdb",
                "skippedRanks": [
                    3,
                    4
                ]
            },
            {
                "operation": "Query + output projections",
                "parts": [
                    "query",
                    "attention-output"
                ],
                "operator": "aten::mm",
                "profileMs": 446.652,
                "writers": [
                    "Memory",
                    "Scheduling",
                    "Compute",
                    "Registers",
                    "Fusion"
                ],
                "gpuTests": [
                    4,
                    2
                ],
                "batches": [
                    [
                        {
                            "id": "4033c8066",
                            "rank": 1,
                            "gpu": 0,
                            "specialty": "Memory",
                            "outcome": "passed",
                            "localRatio": 1.255218,
                            "modelRatio": 1.005482
                        },
                        {
                            "id": "a3214b2f0",
                            "rank": 2,
                            "gpu": 1,
                            "specialty": "Memory",
                            "outcome": "speed",
                            "localRatio": 1.209155,
                            "modelRatio": 0.99538
                        }
                    ]
                ],
                "visitWinner": "4033c8066",
                "skippedRanks": [
                    3,
                    4
                ]
            },
            {
                "operation": "Output head",
                "parts": [
                    "output-head"
                ],
                "operator": "aten::mm",
                "profileMs": 174.725,
                "writers": [
                    "Memory",
                    "Scheduling"
                ],
                "gpuTests": [
                    3,
                    1
                ],
                "batches": [
                    [
                        {
                            "id": "5ebd77b0b",
                            "rank": 1,
                            "gpu": 0,
                            "specialty": "Memory",
                            "outcome": "speed",
                            "localRatio": 1.159308,
                            "modelRatio": 0.98656
                        },
                        {
                            "id": "a2edf5aad",
                            "rank": 2,
                            "gpu": 1,
                            "specialty": "Scheduling",
                            "outcome": "passed",
                            "localRatio": 1.157671,
                            "modelRatio": 1.007231
                        }
                    ]
                ],
                "visitWinner": "a2edf5aad",
                "skippedRanks": [
                    3,
                    4
                ]
            },
            {
                "operation": "Key + value projections",
                "parts": [
                    "key-value"
                ],
                "operator": "aten::mm",
                "profileMs": 151.241,
                "writers": [
                    "Memory",
                    "Scheduling",
                    "Compute",
                    "Registers"
                ],
                "gpuTests": [
                    2,
                    2
                ],
                "batches": [
                    [
                        {
                            "id": "77ba3ab26",
                            "rank": 1,
                            "gpu": 0,
                            "specialty": "Scheduling",
                            "outcome": "correctness",
                            "localRatio": 1.151961,
                            "modelRatio": null
                        },
                        {
                            "id": "cab906273",
                            "rank": 2,
                            "gpu": 1,
                            "specialty": "Compute",
                            "outcome": "passed",
                            "localRatio": 1.136979,
                            "modelRatio": 1.006624
                        }
                    ]
                ],
                "visitWinner": "cab906273",
                "skippedRanks": []
            }
        ],
        "peakWriters": 5
    }
};
