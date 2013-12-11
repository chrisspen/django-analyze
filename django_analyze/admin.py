from django.contrib import admin

class PredictorAdmin(admin.ModelAdmin):
    
    list_display = (
        'id',
        'algorithm',
        'trained_datetime',
        'testing_mean_absolute_error_str',
        'training_seconds',
        'fresh',
    )
    list_filter = (
        'fresh',
    )
    readonly_fields = (
        'testing_mean_absolute_error_str',
    )
    
    def testing_mean_absolute_error_str(self, obj):
        if not obj or obj.testing_mean_absolute_error is None:
            return ''
        return '%.4f' % obj.testing_mean_absolute_error
    testing_mean_absolute_error_str.short_description = 'testing mean absolute error'
    testing_mean_absolute_error_str.admin_order_field = 'testing_mean_absolute_error'