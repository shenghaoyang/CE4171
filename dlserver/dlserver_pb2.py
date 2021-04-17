# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: dlserver.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='dlserver.proto',
  package='info.shenghaoyang.ntu.ntu_iot.dlserver',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x0e\x64lserver.proto\x12&info.shenghaoyang.ntu.ntu_iot.dlserver\x1a\x1bgoogle/protobuf/empty.proto\"-\n\x10InferenceRequest\x12\x19\n\raudio_samples\x18\x01 \x03(\x02\x42\x02\x10\x01\"Q\n\x11InferenceResponse\x12<\n\x05label\x18\x01 \x01(\x0e\x32-.info.shenghaoyang.ntu.ntu_iot.dlserver.Label\"j\n\x0fTrainingRequest\x12<\n\x05label\x18\x01 \x01(\x0e\x32-.info.shenghaoyang.ntu.ntu_iot.dlserver.Label\x12\x19\n\raudio_samples\x18\x02 \x03(\x02\x42\x02\x10\x01*Q\n\x05Label\x12\x08\n\x04\x44OWN\x10\x00\x12\x06\n\x02GO\x10\x01\x12\x08\n\x04LEFT\x10\x02\x12\x06\n\x02NO\x10\x03\x12\t\n\x05RIGHT\x10\x04\x12\x08\n\x04STOP\x10\x05\x12\x06\n\x02UP\x10\x06\x12\x07\n\x03YES\x10\x07\x32\xe6\x01\n\x08\x44LServer\x12~\n\x05Infer\x12\x38.info.shenghaoyang.ntu.ntu_iot.dlserver.InferenceRequest\x1a\x39.info.shenghaoyang.ntu.ntu_iot.dlserver.InferenceResponse\"\x00\x12Z\n\x05Train\x12\x37.info.shenghaoyang.ntu.ntu_iot.dlserver.TrainingRequest\x1a\x16.google.protobuf.Empty\"\x00\x62\x06proto3'
  ,
  dependencies=[google_dot_protobuf_dot_empty__pb2.DESCRIPTOR,])

_LABEL = _descriptor.EnumDescriptor(
  name='Label',
  full_name='info.shenghaoyang.ntu.ntu_iot.dlserver.Label',
  filename=None,
  file=DESCRIPTOR,
  create_key=_descriptor._internal_create_key,
  values=[
    _descriptor.EnumValueDescriptor(
      name='DOWN', index=0, number=0,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='GO', index=1, number=1,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='LEFT', index=2, number=2,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='NO', index=3, number=3,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='RIGHT', index=4, number=4,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='STOP', index=5, number=5,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='UP', index=6, number=6,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='YES', index=7, number=7,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=325,
  serialized_end=406,
)
_sym_db.RegisterEnumDescriptor(_LABEL)

Label = enum_type_wrapper.EnumTypeWrapper(_LABEL)
DOWN = 0
GO = 1
LEFT = 2
NO = 3
RIGHT = 4
STOP = 5
UP = 6
YES = 7



_INFERENCEREQUEST = _descriptor.Descriptor(
  name='InferenceRequest',
  full_name='info.shenghaoyang.ntu.ntu_iot.dlserver.InferenceRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='audio_samples', full_name='info.shenghaoyang.ntu.ntu_iot.dlserver.InferenceRequest.audio_samples', index=0,
      number=1, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=b'\020\001', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=87,
  serialized_end=132,
)


_INFERENCERESPONSE = _descriptor.Descriptor(
  name='InferenceResponse',
  full_name='info.shenghaoyang.ntu.ntu_iot.dlserver.InferenceResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='label', full_name='info.shenghaoyang.ntu.ntu_iot.dlserver.InferenceResponse.label', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=134,
  serialized_end=215,
)


_TRAININGREQUEST = _descriptor.Descriptor(
  name='TrainingRequest',
  full_name='info.shenghaoyang.ntu.ntu_iot.dlserver.TrainingRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='label', full_name='info.shenghaoyang.ntu.ntu_iot.dlserver.TrainingRequest.label', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='audio_samples', full_name='info.shenghaoyang.ntu.ntu_iot.dlserver.TrainingRequest.audio_samples', index=1,
      number=2, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=b'\020\001', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=217,
  serialized_end=323,
)

_INFERENCERESPONSE.fields_by_name['label'].enum_type = _LABEL
_TRAININGREQUEST.fields_by_name['label'].enum_type = _LABEL
DESCRIPTOR.message_types_by_name['InferenceRequest'] = _INFERENCEREQUEST
DESCRIPTOR.message_types_by_name['InferenceResponse'] = _INFERENCERESPONSE
DESCRIPTOR.message_types_by_name['TrainingRequest'] = _TRAININGREQUEST
DESCRIPTOR.enum_types_by_name['Label'] = _LABEL
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

InferenceRequest = _reflection.GeneratedProtocolMessageType('InferenceRequest', (_message.Message,), {
  'DESCRIPTOR' : _INFERENCEREQUEST,
  '__module__' : 'dlserver_pb2'
  # @@protoc_insertion_point(class_scope:info.shenghaoyang.ntu.ntu_iot.dlserver.InferenceRequest)
  })
_sym_db.RegisterMessage(InferenceRequest)

InferenceResponse = _reflection.GeneratedProtocolMessageType('InferenceResponse', (_message.Message,), {
  'DESCRIPTOR' : _INFERENCERESPONSE,
  '__module__' : 'dlserver_pb2'
  # @@protoc_insertion_point(class_scope:info.shenghaoyang.ntu.ntu_iot.dlserver.InferenceResponse)
  })
_sym_db.RegisterMessage(InferenceResponse)

TrainingRequest = _reflection.GeneratedProtocolMessageType('TrainingRequest', (_message.Message,), {
  'DESCRIPTOR' : _TRAININGREQUEST,
  '__module__' : 'dlserver_pb2'
  # @@protoc_insertion_point(class_scope:info.shenghaoyang.ntu.ntu_iot.dlserver.TrainingRequest)
  })
_sym_db.RegisterMessage(TrainingRequest)


_INFERENCEREQUEST.fields_by_name['audio_samples']._options = None
_TRAININGREQUEST.fields_by_name['audio_samples']._options = None

_DLSERVER = _descriptor.ServiceDescriptor(
  name='DLServer',
  full_name='info.shenghaoyang.ntu.ntu_iot.dlserver.DLServer',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=409,
  serialized_end=639,
  methods=[
  _descriptor.MethodDescriptor(
    name='Infer',
    full_name='info.shenghaoyang.ntu.ntu_iot.dlserver.DLServer.Infer',
    index=0,
    containing_service=None,
    input_type=_INFERENCEREQUEST,
    output_type=_INFERENCERESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='Train',
    full_name='info.shenghaoyang.ntu.ntu_iot.dlserver.DLServer.Train',
    index=1,
    containing_service=None,
    input_type=_TRAININGREQUEST,
    output_type=google_dot_protobuf_dot_empty__pb2._EMPTY,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_DLSERVER)

DESCRIPTOR.services_by_name['DLServer'] = _DLSERVER

# @@protoc_insertion_point(module_scope)