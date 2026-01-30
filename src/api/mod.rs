//! API module - REST and gRPC handlers

pub mod rest;
pub mod grpc;
pub mod dto;

pub use rest::create_rest_router;
