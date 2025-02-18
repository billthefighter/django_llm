from django.contrib import admin
from django.utils.html import format_html
from django.db.models import Sum
from .models import LLMProvider, ChainExecution, ChainStep, TokenUsageLog, StoredArtifact, ChainStepSequence, ChainStepDependency, ChainStepTemplate
from .models.llmaestro import DjangoTask, DjangoSubTask, DjangoTokenUsage, DjangoAgentConfig

@admin.register(LLMProvider)
class LLMProviderAdmin(admin.ModelAdmin):
    list_display = ('name', 'model', 'is_active', 'total_usage', 'total_cost', 'created_at')
    list_filter = ('is_active', 'name', 'model')
    search_fields = ('name', 'model')
    readonly_fields = ('created_at', 'updated_at')
    
    def total_usage(self, obj):
        total = obj.usage_logs.aggregate(Sum('total_tokens'))['total_tokens__sum']
        return f"{total:,}" if total else "0"
    total_usage.short_description = 'Total Tokens'
    
    def total_cost(self, obj):
        total = obj.usage_logs.aggregate(Sum('estimated_cost'))['estimated_cost__sum']
        return f"${total:.2f}" if total else "$0.00"
    total_cost.short_description = 'Total Cost'

@admin.register(ChainExecution)
class ChainExecutionAdmin(admin.ModelAdmin):
    list_display = ('id', 'chain_type', 'status', 'started_at', 'completed_at', 'duration', 'error_status')
    list_filter = ('status', 'chain_type', 'started_at')
    search_fields = ('chain_type', 'error_message')
    readonly_fields = ('started_at', 'completed_at')
    
    def duration(self, obj):
        if obj.completed_at and obj.started_at:
            duration = obj.completed_at - obj.started_at
            return f"{duration.total_seconds():.2f}s"
        return "-"
    duration.short_description = 'Duration'
    
    def error_status(self, obj):
        if obj.error_message:
            return format_html(
                '<span style="color: red;">✘</span>'
            )
        return format_html(
            '<span style="color: green;">✓</span>'
        )
    error_status.short_description = 'Error'

@admin.register(ChainStepSequence)
class ChainStepSequenceAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'chain_link', 'created_at')
    list_filter = ('created_at', 'name')
    search_fields = ('name', 'description', 'chain_execution__chain_type')
    
    def chain_link(self, obj):
        return format_html(
            '<a href="{}">{}</a>',
            f'../chainexecution/{obj.chain_execution.id}',
            f'{obj.chain_execution.chain_type} ({obj.chain_execution.id})'
        )
    chain_link.short_description = 'Chain'

@admin.register(ChainStepDependency)
class ChainStepDependencyAdmin(admin.ModelAdmin):
    list_display = ('id', 'sequence_link', 'from_step_link', 'to_step_link', 'created_at')
    list_filter = ('created_at',)
    search_fields = ('sequence__name', 'from_step__step_type', 'to_step__step_type')
    
    def sequence_link(self, obj):
        return format_html(
            '<a href="{}">{}</a>',
            f'../chainstepsequence/{obj.sequence.id}',
            f'{obj.sequence.name}'
        )
    sequence_link.short_description = 'Sequence'
    
    def from_step_link(self, obj):
        return format_html(
            '<a href="{}">{}</a>',
            f'../chainstep/{obj.from_step.id}',
            f'{obj.from_step.step_type}'
        )
    from_step_link.short_description = 'From Step'
    
    def to_step_link(self, obj):
        return format_html(
            '<a href="{}">{}</a>',
            f'../chainstep/{obj.to_step.id}',
            f'{obj.to_step.step_type}'
        )
    to_step_link.short_description = 'To Step'

@admin.register(ChainStepTemplate)
class ChainStepTemplateAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'created_at', 'updated_at')
    list_filter = ('created_at', 'updated_at')
    search_fields = ('name', 'description')
    readonly_fields = ('created_at', 'updated_at')

@admin.register(ChainStep)
class ChainStepAdmin(admin.ModelAdmin):
    list_display = ('id', 'chain_link', 'sequence_link', 'template_link', 'step_type', 'duration', 'error_status')
    list_filter = ('step_type', 'started_at')
    search_fields = ('step_type', 'error_message', 'chain_execution__chain_type')
    readonly_fields = ('started_at', 'completed_at')
    
    def sequence_link(self, obj):
        if obj.sequence:
            return format_html(
                '<a href="{}">{}</a>',
                f'../chainstepsequence/{obj.sequence.id}',
                f'{obj.sequence.name}'
            )
        return "-"
    sequence_link.short_description = 'Sequence'
    
    def template_link(self, obj):
        if obj.template:
            return format_html(
                '<a href="{}">{}</a>',
                f'../chainsteptemplate/{obj.template.id}',
                f'{obj.template.name}'
            )
        return "-"
    template_link.short_description = 'Template'
    
    def chain_link(self, obj):
        return format_html(
            '<a href="{}">{}</a>',
            f'../chainexecution/{obj.chain_execution.id}',
            f'{obj.chain_execution.chain_type} ({obj.chain_execution.id})'
        )
    chain_link.short_description = 'Chain'
    
    def duration(self, obj):
        if obj.completed_at and obj.started_at:
            duration = obj.completed_at - obj.started_at
            return f"{duration.total_seconds():.2f}s"
        return "-"
    duration.short_description = 'Duration'
    
    def error_status(self, obj):
        if obj.error_message:
            return format_html(
                '<span style="color: red;">✘</span>'
            )
        return format_html(
            '<span style="color: green;">✓</span>'
        )
    error_status.short_description = 'Error'

@admin.register(TokenUsageLog)
class TokenUsageLogAdmin(admin.ModelAdmin):
    list_display = ('id', 'provider_name', 'chain_step_link', 'total_tokens', 'estimated_cost', 'timestamp')
    list_filter = ('provider__name', 'timestamp', 'provider__model')
    search_fields = ('chain_step__step_type', 'provider__name')
    
    def provider_name(self, obj):
        return f"{obj.provider.name} ({obj.provider.model})"
    provider_name.short_description = 'Provider'
    
    def chain_step_link(self, obj):
        return format_html(
            '<a href="{}">{}</a>',
            f'../chainstep/{obj.chain_step.id}',
            f'{obj.chain_step.step_type} ({obj.chain_step.id})'
        )
    chain_step_link.short_description = 'Chain Step'

@admin.register(StoredArtifact)
class StoredArtifactAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'chain_link', 'step_link', 'created_at')
    list_filter = ('created_at', 'name')
    search_fields = ('name', 'chain_execution__chain_type', 'step__step_type')
    
    def chain_link(self, obj):
        return format_html(
            '<a href="{}">{}</a>',
            f'../chainexecution/{obj.chain_execution.id}',
            f'{obj.chain_execution.chain_type} ({obj.chain_execution.id})'
        )
    chain_link.short_description = 'Chain'
    
    def step_link(self, obj):
        if obj.step:
            return format_html(
                '<a href="{}">{}</a>',
                f'../chainstep/{obj.step.id}',
                f'{obj.step.step_type} ({obj.step.id})'
            )
        return "-"
    step_link.short_description = 'Step'

@admin.register(DjangoTask)
class TaskAdmin(admin.ModelAdmin):
    list_display = ['task_id', 'type', 'status', 'created_at', 'updated_at']
    search_fields = ['task_id', 'type']
    list_filter = ['status', 'type']

@admin.register(DjangoSubTask)
class SubTaskAdmin(admin.ModelAdmin):
    list_display = ['subtask_id', 'type', 'status', 'parent_task', 'created_at']
    list_filter = ['status', 'type']
    search_fields = ['subtask_id']

@admin.register(DjangoAgentConfig)
class AgentConfigAdmin(admin.ModelAdmin):
    list_display = ['name', 'is_active', 'created_at', 'updated_at']
    list_filter = ['is_active'] 