# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: predictor.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()

DESCRIPTOR = _descriptor.FileDescriptor(
    name='predictor.proto',
    package='',
    syntax='proto2',
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
    serialized_pb=b'\n\x0fpredictor.proto\"o\n\x0ePredictRequest\x12\x0b\n\x03msg\x18\x01 \x02(\t\x12\x17\n\x0cn_prediction\x18\x02 \x01(\x03:\x01\x35\x12 \n\x13time_offset_seconds\x18\x03 \x01(\x03:\x03\x33\x30\x30\x12\x15\n\nsim_cutoff\x18\x04 \x01(\x02:\x01\x30\"l\n\x0cPredictReply\x12.\n\x06result\x18\x01 \x03(\x0b\x32\x1e.PredictReply.PredictReplyElem\x1a,\n\x10PredictReplyElem\x12\x0b\n\x03msg\x18\x01 \x02(\t\x12\x0b\n\x03sim\x18\x02 \x02(\x02\x32?\n\rChatPredictor\x12.\n\nPredictOne\x12\x0f.PredictRequest\x1a\r.PredictReply\"\x00'
)

_PREDICTREQUEST = _descriptor.Descriptor(
    name='PredictRequest',
    full_name='PredictRequest',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name='msg', full_name='PredictRequest.msg', index=0,
            number=1, type=9, cpp_type=9, label=2,
            has_default_value=False, default_value=b"".decode('utf-8'),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='n_prediction', full_name='PredictRequest.n_prediction', index=1,
            number=2, type=3, cpp_type=2, label=1,
            has_default_value=True, default_value=5,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='time_offset_seconds', full_name='PredictRequest.time_offset_seconds', index=2,
            number=3, type=3, cpp_type=2, label=1,
            has_default_value=True, default_value=300,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='sim_cutoff', full_name='PredictRequest.sim_cutoff', index=3,
            number=4, type=2, cpp_type=6, label=1,
            has_default_value=True, default_value=float(0),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=False,
    syntax='proto2',
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=19,
    serialized_end=130,
)

_PREDICTREPLY_PREDICTREPLYELEM = _descriptor.Descriptor(
    name='PredictReplyElem',
    full_name='PredictReply.PredictReplyElem',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name='msg', full_name='PredictReply.PredictReplyElem.msg', index=0,
            number=1, type=9, cpp_type=9, label=2,
            has_default_value=False, default_value=b"".decode('utf-8'),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='sim', full_name='PredictReply.PredictReplyElem.sim', index=1,
            number=2, type=2, cpp_type=6, label=2,
            has_default_value=False, default_value=float(0),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=False,
    syntax='proto2',
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=196,
    serialized_end=240,
)

_PREDICTREPLY = _descriptor.Descriptor(
    name='PredictReply',
    full_name='PredictReply',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name='result', full_name='PredictReply.result', index=0,
            number=1, type=11, cpp_type=10, label=3,
            has_default_value=False, default_value=[],
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
    ],
    extensions=[
    ],
    nested_types=[_PREDICTREPLY_PREDICTREPLYELEM, ],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=False,
    syntax='proto2',
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=132,
    serialized_end=240,
)

_PREDICTREPLY_PREDICTREPLYELEM.containing_type = _PREDICTREPLY
_PREDICTREPLY.fields_by_name['result'].message_type = _PREDICTREPLY_PREDICTREPLYELEM
DESCRIPTOR.message_types_by_name['PredictRequest'] = _PREDICTREQUEST
DESCRIPTOR.message_types_by_name['PredictReply'] = _PREDICTREPLY
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

PredictRequest = _reflection.GeneratedProtocolMessageType('PredictRequest', (_message.Message,), {
    'DESCRIPTOR': _PREDICTREQUEST,
    '__module__': 'predictor_pb2'
    # @@protoc_insertion_point(class_scope:PredictRequest)
})
_sym_db.RegisterMessage(PredictRequest)

PredictReply = _reflection.GeneratedProtocolMessageType('PredictReply', (_message.Message,), {

    'PredictReplyElem': _reflection.GeneratedProtocolMessageType('PredictReplyElem', (_message.Message,), {
        'DESCRIPTOR': _PREDICTREPLY_PREDICTREPLYELEM,
        '__module__': 'predictor_pb2'
        # @@protoc_insertion_point(class_scope:PredictReply.PredictReplyElem)
    })
    ,
    'DESCRIPTOR': _PREDICTREPLY,
    '__module__': 'predictor_pb2'
    # @@protoc_insertion_point(class_scope:PredictReply)
})
_sym_db.RegisterMessage(PredictReply)
_sym_db.RegisterMessage(PredictReply.PredictReplyElem)

_CHATPREDICTOR = _descriptor.ServiceDescriptor(
    name='ChatPredictor',
    full_name='ChatPredictor',
    file=DESCRIPTOR,
    index=0,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
    serialized_start=242,
    serialized_end=305,
    methods=[
        _descriptor.MethodDescriptor(
            name='PredictOne',
            full_name='ChatPredictor.PredictOne',
            index=0,
            containing_service=None,
            input_type=_PREDICTREQUEST,
            output_type=_PREDICTREPLY,
            serialized_options=None,
            create_key=_descriptor._internal_create_key,
        ),
    ])
_sym_db.RegisterServiceDescriptor(_CHATPREDICTOR)

DESCRIPTOR.services_by_name['ChatPredictor'] = _CHATPREDICTOR

# @@protoc_insertion_point(module_scope)
