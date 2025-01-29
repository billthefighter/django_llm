from django.contrib import admin
from django.utils.html import format_html
from django.db.models import Sum
from .models import LLMProvider, ChainExecution, ChainStep, TokenUsageLog, StoredArtifact

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

@admin.register(ChainStep)
class ChainStepAdmin(admin.ModelAdmin):
    list_display = ('id', 'chain_link', 'step_type', 'order', 'duration', 'error_status')
    list_filter = ('step_type', 'started_at')
    search_fields = ('step_type', 'error_message', 'chain_execution__chain_type')
    readonly_fields = ('started_at', 'completed_at')
    
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