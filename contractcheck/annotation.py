"""Annotation semantics shared by standalone checks and dataset audit."""
from . import common


def validate_annotation(doc, document, result, *, item=None, ontology=None):
    common.check_utc_timestamp(result, doc['created_at'], document, '/created_at')
    if 'review' in doc:
        stamp = doc['review']['reviewed_at']
        common.check_utc_timestamp(result, stamp, document, '/review/reviewed_at')
        common.check_time_order(result, doc['created_at'], stamp, document, '/review/reviewed_at')
    common.check_unique(result, ((o['id'], f'/objects/{i}/id') for i, o in enumerate(doc['objects'])),
                        document, 'annotation', 'object id')
    if doc.get('supersedes') == doc['id']:
        result.add_error(document, '/supersedes', 'annotation_cycle', 'annotation cannot supersede itself')
    if item is not None:
        if doc['status'] != 'approved':
            result.add_error(document, '/status', 'annotation_not_approved', 'released annotations must be approved')
        for field, expected in [('id', item['annotation']['id']), ('item_id', item['id']),
                                ('image_sha256', item['image']['sha256'])]:
            if doc[field] != expected:
                result.add_error(document, '/' + field, 'annotation_mismatch', 'annotation does not match dataset item')
    if ontology is not None:
        for field, expected in [('ontology_id', ontology['id']), ('ontology_version', ontology['version'])]:
            if doc[field] != expected:
                result.add_error(document, '/' + field, 'annotation_mismatch', 'ontology identity differs')
    for i, obj in enumerate(doc['objects']):
        if ontology is not None and obj['class_id'] not in {c['id'] for c in ontology['classes']}:
            result.add_error(document, f'/objects/{i}/class_id', 'unknown_class', 'class does not exist in ontology')
        common.check_box(result, obj['bbox'], item['width'] if item else float('inf'),
                         item['height'] if item else float('inf'), document, f'/objects/{i}/bbox')


def check_classes(classes, document, result, prefix='/classes'):
    for key in ('id', 'label'):
        common.check_unique(result, ((c[key], f'{prefix}/{i}/{key}') for i, c in enumerate(classes)),
                            document, 'ontology', f'class {key}')
