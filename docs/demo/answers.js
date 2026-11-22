// Real output from saved benchmark token IDs, decoded with each pinned tokenizer.
// Each model produced identical greedy token IDs on all three paths.
const recordedAnswers = {
    "prompt": "Explain the theory of general relativity in detail, covering spacetime curvature, the equivalence principle, and gravitational waves.",
    "models": {
        "mistral": {
            "text": "\n\nGeneral Relativity (GR) is a theory of gravitation that was developed by Albert Einstein between 1907 and 1915. It is a theory of gravitation based on the principle that the observed gravitational effect between masses results from their warping of spacetime.\n\n1. Spacetime Curvature:\n\nIn GR, space and time are not separate entities but are combined into a four-dimensional structure called spacetime. The presence of mass and energy causes spacetime to curve, and this curvature is what we perceive as gravity. This concept is known",
            "tokenEnds": [
                1, 2, 9, 13, 20, 22, 24, 25, 28, 30, 37, 40, 45, 52, 57, 61,
                71, 74, 81, 90, 98, 99, 100, 101, 102, 103, 107, 108, 109, 110, 111, 112,
                113, 116, 119, 121, 128, 131, 136, 143, 149, 152, 156, 166, 171, 175, 184, 189,
                198, 205, 213, 220, 228, 233, 239, 243, 247, 250, 253, 255, 260, 261, 262, 263,
                264, 265, 268, 270, 275, 279, 280, 285, 286, 287, 288, 290, 293, 294, 300, 304,
                309, 313, 317, 326, 335, 339, 343, 352, 357, 359, 364, 365, 376, 386, 393, 396,
                398, 403, 404, 408, 417, 420, 425, 429, 436, 443, 446, 448, 453, 456, 462, 463,
                467, 472, 477, 482, 485, 490, 493, 497, 502, 505, 513, 514, 519, 527, 530, 536,
            ],
        },
        "qwen": {
            "text": " The theory of general relativity, formulated by Albert Einstein in 1915, is a fundamental theory in physics that describes the force of gravity and the structure of spacetime. It is a geometric theory of gravitation, which means that it describes gravity not as a force, but as a consequence of the curvature of spacetime caused by the presence of mass and energy.\n\n### Spacetime Curvature\n\nIn general relativity, spacetime is a four-dimensional continuum that combines the three dimensions of space (length, width, and height) with the fourth dimension of time. This four-dimensional space is called spacetime. The presence",
            "tokenEnds": [
                4, 11, 14, 22, 26, 33, 34, 45, 48, 55, 64, 67, 68, 69, 70, 71,
                72, 73, 76, 78, 90, 97, 100, 108, 113, 123, 127, 133, 136, 144, 148, 152,
                162, 165, 170, 175, 176, 179, 182, 184, 194, 201, 204, 209, 216, 217, 223, 229,
                234, 237, 247, 255, 259, 262, 264, 270, 271, 275, 278, 280, 292, 295, 299, 309,
                312, 317, 322, 329, 332, 336, 345, 348, 353, 357, 364, 367, 370, 373, 375, 380,
                384, 390, 392, 394, 402, 406, 413, 414, 419, 424, 427, 429, 434, 446, 456, 461,
                470, 474, 480, 491, 494, 500, 502, 508, 509, 515, 516, 520, 527, 528, 533, 537,
                544, 554, 557, 562, 563, 568, 573, 585, 591, 594, 601, 606, 611, 612, 616, 625,
            ],
        },
        "llama": {
            "text": " Describe the history of the theory, including the work of Einstein, Schwarzschild, and others. Discuss the experimental evidence for general relativity, including the precession of Mercury’s perihelion, the deflection of light by the Sun, and the gravitational redshift of light. Explain the concept of black holes, including their formation, properties, and potential detection. Discuss the concept of gravitational waves, including their production, detection, and potential applications. Describe the current state of research in general relativity, including ongoing experiments and future directions.\nGeneral relativity is a theory of gravitation that was developed by Albert Einstein in 191",
            "tokenEnds": [
                9, 13, 21, 24, 28, 35, 36, 46, 50, 55, 58, 67, 68, 75, 77, 82,
                83, 87, 94, 95, 103, 107, 120, 129, 133, 141, 145, 152, 153, 163, 167, 171,
                178, 181, 189, 191, 195, 197, 199, 202, 203, 207, 211, 218, 221, 227, 230, 234,
                238, 239, 243, 247, 261, 265, 270, 273, 279, 280, 288, 292, 300, 303, 309, 315,
                316, 326, 332, 342, 343, 354, 355, 359, 369, 379, 380, 388, 392, 400, 403, 417,
                423, 424, 434, 440, 451, 452, 462, 463, 467, 477, 490, 491, 500, 504, 512, 518,
                521, 530, 533, 541, 545, 552, 553, 563, 571, 583, 587, 594, 605, 607, 614, 618,
                625, 628, 630, 637, 640, 645, 652, 657, 661, 671, 674, 681, 690, 693, 694, 697,
            ],
        },
    },
};
