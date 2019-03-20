from collections import namedtuple

Region = namedtuple('Region', ['scene', 'video', 'rid', 'anchor', 'size', 'start', 'end'])

regions_deathCircle_v1 = [
    # Region 1
    Region('deathCircle', 1, 1, (288, 558), (256, 256), 0, 9600),
    # Region('deathCircle', 1, 1, (288, 558), (256, 256), 9900, 13950),

    # Region 2
    # Region('deathCircle', 1, 2, (643, 393), (256, 256), 0, 9600),
    # Region('deathCircle', 1, 2, (643, 393), (256, 256), 9900, 13950),

    # # Region 3
    # Region('deathCircle', 1, 3, (705, 972), (256, 256), 0, 9600),
    # Region('deathCircle', 1, 3, (705, 972), (256, 256), 9900, 13950),

    # # Region 4
    # Region('deathCircle', 1, 4, (380, 1068), (256, 256), 0, 9600),
    # Region('deathCircle', 1, 4, (380, 1068), (256, 256), 9900, 13950),

    # # Region 5
    # Region('deathCircle', 1, 5, (497, 1266), (256, 256), 0, 9600),
    # Region('deathCircle', 1, 5, (497, 1266), (256, 256), 9900, 13950)
]

test = [
    dict(scene='deathCircle', video=3, rid=1, anchor=(186, 356), size=(256, 256), seq=[(0, 300)])
]
regions_deathCircle_v3 = [
    # Region 1
    dict(scene='deathCircle', video=3, rid=1, anchor=(186, 356), size=(256, 256),
         seq=[(550, 2000),
              (2500, 3600),
              (4950, 5234),
              (5285, 9000),
              (10100, 12491)]),

    # Region 2
    dict(scene='deathCircle', video=3, rid=2, anchor=(620, 220), size=(256, 256),
         seq=[(550, 2000),
              (2500, 3600),
              (4950, 5234),
              (5285, 9000),
              (10100, 12491)]),

    # Region 3
    dict(scene='deathCircle', video=3, rid=3, anchor=(743, 905), size=(256, 256),
         seq=[(550, 2000),
              (2500, 3600),
              (4950, 5234),
              (5285, 9000),
              (10100, 12491)]),

    # Region 4
    dict(scene='deathCircle', video=3, rid=4, anchor=(315, 1111), size=(256, 256),
         seq=[(550, 2000),
              (2500, 3600),
              (4950, 5234),
              (5285, 9000),
              (10100, 12491)]),

    # Region 5
    dict(scene='deathCircle', video=3, rid=5, anchor=(454, 1309), size=(256, 256),
         seq=[(550, 2000),
              (2500, 3600),
              (4950, 5234),
              (5285, 9000),
              (10100, 12491)])
]
