"""Django-specific orchestration layer for managing LLM conversations with persistence."""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Callable, Awaitable, Type, cast
from uuid import uuid4

from django.db import transaction
from django.db.models import Model

from llmaestro.core.orchestrator import Orchestrator, ExecutionMetadata
from llmaestro.core.conversations import ConversationGraph, ConversationNode, ConversationEdge
from llmaestro.core.models import LLMResponse
from llmaestro.prompts.base import BasePrompt
from llmaestro.agents.agent_pool import AgentPool

from django_llm.models.models import (
    DjangoConversationGraph,
    DjangoConversationNode,
    DjangoConversationEdge,
    DjangoExecutionMetadata,
)
from django_llm.orchestration.model_mapper import ModelMapper


class DjangoOrchestrator(Orchestrator):
    """
    Django-specific orchestrator that extends LLMaestro's Orchestrator with persistence.
    
    This class handles the mapping between Pydantic objects and Django models,
    providing seamless persistence for LLMaestro conversations and execution metadata.
    """

    def __init__(self, agent_pool: AgentPool):
        """
        Initialize the Django orchestrator.
        
        Args:
            agent_pool: The agent pool to use for executing prompts
        """
        super().__init__(agent_pool)
        # Use Dict[str, Any] to avoid type errors with Django models
        self.django_conversations: Dict[str, Any] = {}

    @transaction.atomic
    async def create_conversation(
        self, name: str, initial_prompt: BasePrompt, metadata: Optional[Dict[str, Any]] = None
    ) -> ConversationGraph:
        """
        Create a new conversation with an initial prompt and persist it to the database.
        
        Args:
            name: The name of the conversation
            initial_prompt: The initial prompt to start the conversation
            metadata: Optional metadata for the conversation
            
        Returns:
            The created conversation graph
        """
        # Create the conversation in memory first
        conversation = await super().create_conversation(name, initial_prompt, metadata)
        
        # Create the Django model
        django_conversation = DjangoConversationGraph.from_pydantic(conversation, name=name)
        
        # Store the Django model - using Any type to avoid type errors
        self.django_conversations[conversation.id] = django_conversation
        
        # Persist the initial node
        node_id = list(conversation.nodes.keys())[0]
        node = conversation.nodes[node_id]
        await self._persist_node(conversation.id, node)
        
        return conversation

    @transaction.atomic
    async def execute_prompt(
        self,
        conversation: Union[str, ConversationGraph, ConversationNode],
        prompt: BasePrompt,
        dependencies: Optional[List[str]] = None,
        parallel_group: Optional[str] = None,
    ) -> str:
        """
        Execute a prompt and add it to the conversation with persistence.
        
        Args:
            conversation: The conversation to add the prompt to
            prompt: The prompt to execute
            dependencies: Optional list of node IDs that this prompt depends on
            parallel_group: Optional group ID for parallel execution
            
        Returns:
            The ID of the response node
        """
        # Execute the prompt using the parent class
        response_node_id = await super().execute_prompt(
            conversation, prompt, dependencies, parallel_group
        )
        
        # Get the conversation object
        conv_obj = self._get_conversation(conversation)
        
        # Get the prompt node ID (it's the parent of the response node)
        # Convert edges to dict if it's a list
        edges_dict = conv_obj.edges if isinstance(conv_obj.edges, dict) else {edge.id: edge for edge in conv_obj.edges}
        
        for edge in edges_dict.values():
            if edge.target_id == response_node_id and edge.edge_type == "response_to":
                prompt_node_id = edge.source_id
                
                # Persist the prompt node
                await self._persist_node(conv_obj.id, conv_obj.nodes[prompt_node_id])
                
                # Persist the response node
                await self._persist_node(conv_obj.id, conv_obj.nodes[response_node_id])
                
                # Persist the edge
                await self._persist_edge(conv_obj.id, edge)
                break
        
        return response_node_id

    async def _persist_node(self, conversation_id: str, node: ConversationNode) -> None:
        """
        Persist a conversation node to the database.
        
        Args:
            conversation_id: The ID of the conversation
            node: The node to persist
        """
        # Get the Django conversation
        django_conversation = self.django_conversations.get(conversation_id)
        if not django_conversation:
            # Try to load it from the database
            try:
                # Use objects.get() with type ignore to bypass type checking
                django_conversation = DjangoConversationGraph.objects.get(  # type: ignore
                    object_type="ConversationGraph", 
                    data__id=conversation_id
                )
                self.django_conversations[conversation_id] = django_conversation
            except DjangoConversationGraph.DoesNotExist:  # type: ignore
                raise ValueError(f"Conversation {conversation_id} not found in database")
        
        # Create the execution metadata if it exists
        execution_metadata = None
        if "execution" in node.metadata:
            exec_data = node.metadata["execution"]
            execution_metadata = DjangoExecutionMetadata.from_pydantic(
                ExecutionMetadata.model_validate(exec_data)
            )
            execution_metadata.save()  # type: ignore
        
        # Create the Django node
        django_node = DjangoConversationNode.from_pydantic(node)
        
        # Set additional fields - use setattr to bypass type checking
        if execution_metadata:
            setattr(django_node, "execution_metadata", execution_metadata)
        
        # Save the node
        django_node.save()  # type: ignore

    async def _persist_edge(self, conversation_id: str, edge: Any) -> None:
        """
        Persist a conversation edge to the database.
        
        Args:
            conversation_id: The ID of the conversation
            edge: The edge to persist
        """
        # Get the Django conversation
        django_conversation = self.django_conversations.get(conversation_id)
        if not django_conversation:
            # Try to load it from the database
            try:
                # Use objects.get() with type ignore to bypass type checking
                django_conversation = DjangoConversationGraph.objects.get(  # type: ignore
                    object_type="ConversationGraph", 
                    data__id=conversation_id
                )
                self.django_conversations[conversation_id] = django_conversation
            except DjangoConversationGraph.DoesNotExist:  # type: ignore
                raise ValueError(f"Conversation {conversation_id} not found in database")
        
        # Create the Django edge
        django_edge = DjangoConversationEdge.from_pydantic(edge)
        
        # Set the conversation - use setattr to bypass type checking
        setattr(django_edge, "conversation", django_conversation)
        
        # Save the edge
        django_edge.save()  # type: ignore

    @transaction.atomic
    async def load_conversation(self, conversation_id: str) -> ConversationGraph:
        """
        Load a conversation from the database.
        
        Args:
            conversation_id: The ID of the conversation to load
            
        Returns:
            The loaded conversation graph
            
        Raises:
            ValueError: If the conversation is not found
        """
        try:
            # Try to load from the database - use type ignore to bypass type checking
            django_conversation = DjangoConversationGraph.objects.get(  # type: ignore
                object_type="ConversationGraph", 
                data__id=conversation_id
            )
            
            # Convert to Pydantic model
            conversation = django_conversation.to_pydantic()  # type: ignore
            
            # Store in memory
            self.active_conversations[conversation_id] = conversation
            self.django_conversations[conversation_id] = django_conversation
            
            # Set as active
            self.active_conversation_id = conversation_id
            
            return conversation
            
        except DjangoConversationGraph.DoesNotExist:  # type: ignore
            raise ValueError(f"Conversation {conversation_id} not found in database")

    @transaction.atomic
    async def save_conversation(self, conversation_id: Optional[str] = None) -> None:
        """
        Save a conversation to the database.
        
        Args:
            conversation_id: The ID of the conversation to save, or None for the active conversation
            
        Raises:
            ValueError: If the conversation is not found
        """
        # Resolve the conversation ID
        conv_id = self._resolve_conversation_id(conversation_id)
        if not conv_id:
            raise ValueError("No active conversation")
        
        # Get the conversation
        conversation = self.active_conversations.get(conv_id)
        if not conversation:
            raise ValueError(f"Conversation {conv_id} not found")
        
        # Get or create the Django conversation
        django_conversation = self.django_conversations.get(conv_id)
        if not django_conversation:
            django_conversation = DjangoConversationGraph.from_pydantic(conversation)
            # Store the Django model - using Any type to avoid type errors
            self.django_conversations[conv_id] = django_conversation
        else:
            # Update the existing Django conversation
            django_conversation.update_from_pydantic(conversation)  # type: ignore
        
        # Save the Django conversation
        django_conversation.save()  # type: ignore
        
        # Save all nodes and edges
        for node_id, node in conversation.nodes.items():
            await self._persist_node(conv_id, node)
        
        # Convert edges to dict if it's a list
        edges_dict = conversation.edges if isinstance(conversation.edges, dict) else {edge.id: edge for edge in conversation.edges}
        
        for edge_id, edge in edges_dict.items():
            await self._persist_edge(conv_id, edge)

    async def get_all_conversations(self) -> List[ConversationGraph]:
        """
        Get all conversations from the database.
        
        Returns:
            List of all conversation graphs
        """
        # Load all conversations from the database - use type ignore to bypass type checking
        django_conversations = DjangoConversationGraph.objects.all()  # type: ignore
        
        # Convert to Pydantic models
        conversations = []
        for django_conversation in django_conversations:
            conversation = django_conversation.to_pydantic()  # type: ignore
            conversations.append(conversation)
            
            # Store in memory if not already there
            if conversation.id not in self.active_conversations:
                self.active_conversations[conversation.id] = conversation
                self.django_conversations[conversation.id] = django_conversation
        
        return conversations

    async def delete_conversation(self, conversation_id: str) -> None:
        """
        Delete a conversation from the database.
        
        Args:
            conversation_id: The ID of the conversation to delete
            
        Raises:
            ValueError: If the conversation is not found
        """
        # Try to get the Django conversation
        try:
            # Use objects.get() with type ignore to bypass type checking
            django_conversation = DjangoConversationGraph.objects.get(  # type: ignore
                object_type="ConversationGraph", 
                data__id=conversation_id
            )
            
            # Delete from database
            django_conversation.delete()  # type: ignore
            
            # Remove from memory
            self.active_conversations.pop(conversation_id, None)
            self.django_conversations.pop(conversation_id, None)
            
            # Reset active conversation if it was the active one
            if self.active_conversation_id == conversation_id:
                self.active_conversation_id = None
                
        except DjangoConversationGraph.DoesNotExist:  # type: ignore
            raise ValueError(f"Conversation {conversation_id} not found in database")