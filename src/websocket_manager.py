"""
WebSocket Connection Manager for NCS API
Handles real-time communication and broadcasting
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Set, Any, Optional
from fastapi import WebSocket, WebSocketDisconnect
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


class WebSocketConnection:
    """Represents a single WebSocket connection with metadata"""
    
    def __init__(self, websocket: WebSocket, connection_id: Optional[str] = None):
        self.websocket = websocket
        self.connection_id = connection_id or str(uuid.uuid4())
        self.connected_at = datetime.utcnow()
        self.last_ping = time.time()
        self.subscriptions: Set[str] = set()
        self.metadata: Dict[str, Any] = {}
    
    async def send_message(self, message: Dict[str, Any]):
        """Send message to this connection"""
        try:
            await self.websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending message to {self.connection_id}: {e}")
            raise
    
    async def ping(self):
        """Send ping to keep connection alive"""
        try:
            await self.websocket.send_text(json.dumps({"type": "ping", "timestamp": time.time()}))
            self.last_ping = time.time()
        except Exception as e:
            logger.warning(f"Ping failed for connection {self.connection_id}: {e}")
            raise
    
    def add_subscription(self, topic: str):
        """Add subscription to a topic"""
        self.subscriptions.add(topic)
    
    def remove_subscription(self, topic: str):
        """Remove subscription from a topic"""
        self.subscriptions.discard(topic)
    
    def is_subscribed_to(self, topic: str) -> bool:
        """Check if connection is subscribed to a topic"""
        return topic in self.subscriptions


class ConnectionManager:
    """
    WebSocket connection manager with support for:
    - Connection lifecycle management
    - Broadcasting messages
    - Topic-based subscriptions
    - Connection health monitoring
    """
    
    def __init__(self, max_connections: int = 100, ping_interval: int = 30):
        self.connections: Dict[str, WebSocketConnection] = {}
        self.topics: Dict[str, Set[str]] = {}  # topic -> set of connection_ids
        self.max_connections = max_connections
        self.ping_interval = ping_interval
        self._ping_task: Optional[asyncio.Task] = None
        self._start_ping_task()
    
    def _start_ping_task(self):
        """Start background task for keeping connections alive"""
        if self._ping_task is None or self._ping_task.done():
            self._ping_task = asyncio.create_task(self._ping_connections_periodically())
    
    async def connect(self, websocket: WebSocket, connection_id: Optional[str] = None) -> str:
        """
        Accept a new WebSocket connection
        Returns the connection ID
        """
        if len(self.connections) >= self.max_connections:
            await websocket.close(code=1013, reason="Too many connections")
            raise Exception("Maximum connections exceeded")
        
        await websocket.accept()
        
        connection = WebSocketConnection(websocket, connection_id)
        self.connections[connection.connection_id] = connection
        
        logger.info(f"WebSocket connection established: {connection.connection_id}")
        
        # Send welcome message
        await connection.send_message({
            "type": "connected",
            "connection_id": connection.connection_id,
            "timestamp": time.time(),
            "server_info": {
                "version": "1.0.0",
                "features": ["clustering", "real_time_updates", "subscriptions"]
            }
        })
        
        return connection.connection_id
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        # Find connection by websocket object
        connection_id = None
        for cid, conn in self.connections.items():
            if conn.websocket == websocket:
                connection_id = cid
                break
        
        if connection_id:
            self._remove_connection(connection_id)
    
    def disconnect_by_id(self, connection_id: str):
        """Remove a connection by ID"""
        self._remove_connection(connection_id)
    
    def _remove_connection(self, connection_id: str):
        """Internal method to remove a connection"""
        if connection_id in self.connections:
            connection = self.connections[connection_id]
            
            # Remove from all topics
            for topic in list(connection.subscriptions):
                self.unsubscribe(connection_id, topic)
            
            # Remove from connections
            del self.connections[connection_id]
            
            logger.info(f"WebSocket connection removed: {connection_id}")
    
    async def disconnect_all(self):
        """Disconnect all connections"""
        logger.info("Disconnecting all WebSocket connections")
        
        # Cancel ping task
        if self._ping_task and not self._ping_task.done():
            self._ping_task.cancel()
            try:
                await self._ping_task
            except asyncio.CancelledError:
                pass
        
        # Close all connections
        close_tasks = []
        for connection in self.connections.values():
            try:
                close_tasks.append(connection.websocket.close())
            except Exception as e:
                logger.warning(f"Error closing connection {connection.connection_id}: {e}")
        
        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)
        
        self.connections.clear()
        self.topics.clear()
    
    async def send_to_connection(self, connection_id: str, message: Dict[str, Any]) -> bool:
        """
        Send message to specific connection
        Returns True if successful, False if connection not found or error
        """
        if connection_id not in self.connections:
            logger.warning(f"Connection {connection_id} not found")
            return False
        
        try:
            await self.connections[connection_id].send_message(message)
            return True
        except Exception as e:
            logger.error(f"Error sending to connection {connection_id}: {e}")
            self._remove_connection(connection_id)
            return False
    
    async def broadcast(self, message: Dict[str, Any], exclude: Optional[Set[str]] = None):
        """
        Broadcast message to all connected clients
        Optionally exclude specific connection IDs
        """
        if not self.connections:
            return
        
        exclude = exclude or set()
        failed_connections = []
        
        # Add timestamp and message ID
        message_with_meta = {
            **message,
            "timestamp": time.time(),
            "message_id": str(uuid.uuid4())
        }
        
        # Send to all connections
        send_tasks = []
        for connection_id, connection in self.connections.items():
            if connection_id not in exclude:
                send_tasks.append(self._safe_send(connection, message_with_meta, connection_id))
        
        if send_tasks:
            results = await asyncio.gather(*send_tasks, return_exceptions=True)
            
            # Remove failed connections
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    connection_id = list(self.connections.keys())[i]
                    if connection_id not in exclude:
                        failed_connections.append(connection_id)
        
        # Clean up failed connections
        for connection_id in failed_connections:
            self._remove_connection(connection_id)
        
        logger.info(f"Broadcast sent to {len(send_tasks) - len(failed_connections)} connections")
    
    async def _safe_send(self, connection: WebSocketConnection, message: Dict[str, Any], connection_id: str):
        """Safely send message to connection"""
        try:
            await connection.send_message(message)
        except Exception as e:
            logger.warning(f"Failed to send to connection {connection_id}: {e}")
            raise
    
    def subscribe(self, connection_id: str, topic: str) -> bool:
        """Subscribe a connection to a topic"""
        if connection_id not in self.connections:
            return False
        
        # Add to connection's subscriptions
        self.connections[connection_id].add_subscription(topic)
        
        # Add to topic's connections
        if topic not in self.topics:
            self.topics[topic] = set()
        self.topics[topic].add(connection_id)
        
        logger.debug(f"Connection {connection_id} subscribed to topic '{topic}'")
        return True
    
    def unsubscribe(self, connection_id: str, topic: str) -> bool:
        """Unsubscribe a connection from a topic"""
        if connection_id not in self.connections:
            return False
        
        # Remove from connection's subscriptions
        self.connections[connection_id].remove_subscription(topic)
        
        # Remove from topic's connections
        if topic in self.topics:
            self.topics[topic].discard(connection_id)
            
            # Clean up empty topics
            if not self.topics[topic]:
                del self.topics[topic]
        
        logger.debug(f"Connection {connection_id} unsubscribed from topic '{topic}'")
        return True
    
    async def broadcast_to_topic(self, topic: str, message: Dict[str, Any]):
        """Broadcast message to all connections subscribed to a topic"""
        if topic not in self.topics or not self.topics[topic]:
            logger.debug(f"No subscribers for topic '{topic}'")
            return
        
        # Add topic to message
        message_with_topic = {
            **message,
            "topic": topic,
            "timestamp": time.time(),
            "message_id": str(uuid.uuid4())
        }
        
        failed_connections = []
        send_tasks = []
        
        for connection_id in self.topics[topic]:
            if connection_id in self.connections:
                connection = self.connections[connection_id]
                send_tasks.append(self._safe_send(connection, message_with_topic, connection_id))
            else:
                failed_connections.append(connection_id)
        
        # Remove invalid connections from topic
        for connection_id in failed_connections:
            self.topics[topic].discard(connection_id)
        
        if send_tasks:
            results = await asyncio.gather(*send_tasks, return_exceptions=True)
            
            # Handle send failures
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    connection_id = list(self.topics[topic])[i]
                    self._remove_connection(connection_id)
        
        logger.info(f"Topic '{topic}' broadcast sent to {len(send_tasks)} subscribers")
    
    async def _ping_connections_periodically(self):
        """Background task to ping connections and remove stale ones"""
        while True:
            try:
                await asyncio.sleep(self.ping_interval)
                await self._ping_all_connections()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in ping task: {e}")
    
    async def _ping_all_connections(self):
        """Ping all connections and remove unresponsive ones"""
        if not self.connections:
            return
        
        current_time = time.time()
        failed_connections = []
        
        for connection_id, connection in self.connections.items():
            try:
                # Check if connection is stale
                if current_time - connection.last_ping > self.ping_interval * 2:
                    logger.warning(f"Connection {connection_id} appears stale, removing")
                    failed_connections.append(connection_id)
                    continue
                
                # Send ping
                await connection.ping()
                
            except Exception as e:
                logger.warning(f"Ping failed for connection {connection_id}: {e}")
                failed_connections.append(connection_id)
        
        # Remove failed connections
        for connection_id in failed_connections:
            self._remove_connection(connection_id)
        
        if self.connections:
            logger.debug(f"Pinged {len(self.connections)} connections, removed {len(failed_connections)}")
    
    def get_connection_count(self) -> int:
        """Get number of active connections"""
        return len(self.connections)
    
    def get_topic_count(self) -> int:
        """Get number of active topics"""
        return len(self.topics)
    
    def get_connection_info(self, connection_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific connection"""
        if connection_id not in self.connections:
            return None
        
        connection = self.connections[connection_id]
        return {
            "connection_id": connection.connection_id,
            "connected_at": connection.connected_at.isoformat(),
            "last_ping": connection.last_ping,
            "subscriptions": list(connection.subscriptions),
            "metadata": connection.metadata
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection manager statistics"""
        return {
            "active_connections": len(self.connections),
            "active_topics": len(self.topics),
            "max_connections": self.max_connections,
            "ping_interval": self.ping_interval,
            "topic_stats": {
                topic: len(connections) 
                for topic, connections in self.topics.items()
            }
        }